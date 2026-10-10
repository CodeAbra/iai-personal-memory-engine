"""Byte-differential gate for the recall pipeline.

Captures, verbatim, the recall pipeline's intermediate community scores,
ordered seed ids, traversal provenance, deep scoring intermediates (the
normalized candidate pool, the recomputed cosine vector, the lexical T11/T12
flags), the exact-cosine authority merge, the ordered hits/anti_hits, and the
final serialized response bytes -- for a fixed cue set + frozen clock + a
deterministic corpus. It then asserts the capture is DETERMINISTIC (an
old-vs-old self-comparison on current code), SENSITIVE (a perturbed cue or
embedding changes what was captured), and byte-IDENTICAL between the live
reader and a snapshot/alternate-fed reader reloaded from a serialized graph
feed. This is the anti-drift gate every later representation change these
tests gate must pass unchanged; a silent recall drift (correct text, wrong
set/order/score of memories) is the risk this harness exists to catch.

Capture points mirror the real recall path exactly, via test-only
monkeypatch wrappers -- never a production edit:
  - `pipeline._community_gate_max_node_scored` -- (ordered community ids,
    full community_scores dict).
  - `pipeline._collect_graph_pool` + `pipeline._pick_seeds` -- together
    resolve the ordered seed UUID list (`_pick_seeds` returns indices into
    the pool `_collect_graph_pool` just built).
  - `pipeline._normalize_pool` -- the sanitized + L2-normalized candidate
    pool matrix, captured as raw float bits.
  - `pipeline._t11_t12_flags` -- the trigram/FTS lexical boolean flags,
    populated by the live hybrid-Rust scorer path (`use_rust_scorer=True`,
    the same flag `core.dispatch` always requests).
  - `MemoryGraph.two_hop_neighborhood_with_provenance` -- the full
    `{reached_id: (origin_seed, hop, carries_transfer)}` map.
  - `recall_for_response`'s own return -- `hits`/`anti_hits` (order
    significant).
  - `pipeline.merge_authority_hits` -- called directly (a pure function) over
    exact-cosine authority candidates built the same way the production
    dispatch branch builds them, so the captured bundle goes through
    authority inclusion + the token budget-pack before it is serialized.

The cosine score vector is not spied directly: the production matmul is an
inline expression, not a patchable callable. It is instead RECOMPUTED in the
test as a matmul of the captured normalized pool against a cue vector
computed by the same pure functions (`_sanitize_vec` + L2-normalize) the
production path uses on the identical, deterministic cue embedding -- same
inputs, same deterministic arithmetic, same-process float bits.

Both clocks the recall path reads are frozen for the capture: `pipeline._age_penalty`
directly (function-local imports in richclub/session pick it up at call time)
and `pipeline._rank_now` (the Rust-scorer age term), scoped to one capture via
`pytest.MonkeyPatch.context()` so the freeze can never leak across captures.

The adversarial traps below each isolate one rank-index feed source (the
whole-store active scan, the pending second source, recall's seed-centrality
source, and the write-time hook feed) and prove it is both load-bearing (a
committed golden) and non-vacuous (a per-trap perturbation changes the
captured bundle).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pytest

from iai_mcp import graph as graph_mod
from iai_mcp import pipeline as pipeline_mod
from iai_mcp import retrieve
from iai_mcp.pipeline import merge_authority_hits, recall_for_response
from iai_mcp.store import MemoryStore, flush_edge_buffer, flush_record_buffer
from iai_mcp.types import MemoryHit

from tests.rss_rca.corpus import build_corpus, make_record
from tests.rss_rca.phys_footprint_harness import _build_synthetic_edges, _read_record_ids

# Record indices (insertion order) wired with one "contradicts" edge so the
# anti-hit path (`pipeline._find_anti_hits` reads only "contradicts"-typed
# incident edges -- confirmed by direct read) is actually reachable; plain
# hebbian edges alone never produce an anti-hit.
_ANTI_HIT_RECORD_INDEX = 105
_ANTI_HIT_CONTRADICTING_INDEX = 106

_N_RECORDS = 400
_EDGE_FACTOR = 1.3
_SEED = 20260917

_GOLDEN_ROOT = Path(__file__).parent / "fixtures" / "recall_byte_parity"
_GOLDEN_REGEN_ENV = "IAI_MCP_RECALL_BYTE_PARITY_REGEN"

_ADVERSARIAL_CUE = "adversarial spine cue: a distinctive lexical anchor phrase"


@pytest.fixture(autouse=True)
def _crypto_passphrase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake.pop((s, u), None))
    yield fake


class _DeterministicEmbedder:
    """One fixed unit vector per distinct cue TEXT (sha256-seeded, never
    `hash()` -- reproducible across processes, not just within one).
    Different text always yields a different vector; identical text always
    yields the identical vector."""

    def __init__(self, dim: int) -> None:
        self.DIM = dim

    def _vec_for(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.DIM).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed(self, text: str) -> list[float]:
        return self._vec_for(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vec_for(t) for t in texts]


# Fixed cue set covering distinct recall behaviors on the synthetic corpus
# (records read `f"alice daily routine note {i}: she logged a habit and a
# plan"`, `tests/rss_rca/corpus.py::make_record`):
_CUES: dict[str, str] = {
    # Substring of record 5's literal_surface -- exercises the FTS/substring
    # widening path independent of embedding similarity.
    "exact_substring": "alice daily routine note 5",
    # Shares no tokens with any corpus record's surface -- exercises the
    # embedding-only path with zero lexical overlap.
    "paraphrase_only": "distant unrelated topic about shipping manifests and forecasts",
    # Embeddings are synthetic/random per make_record; with ~40 ten-node
    # clusters the top-3 community gate spans multiple communities for any
    # cue whose scores are not degenerate -- this is the required non-empty
    # multi-community cue.
    "multi_community": "alice daily routine note 250",
    # Exact full-surface match on the record wired with a "contradicts" edge
    # (see `_ANTI_HIT_RECORD_INDEX` below) -- the strong FTS/substring boost
    # puts this record in `hits`, making its contradicts-neighbor reachable
    # as an anti-hit.
    "anti_hit_expected": (
        f"alice daily routine note {_ANTI_HIT_RECORD_INDEX}: "
        f"she logged a habit and a plan"
    ),
}


def _seed_store(tmp_path) -> MemoryStore:
    store = MemoryStore(path=tmp_path / "store")
    build_corpus(
        store, n=_N_RECORDS, seed=_SEED, dim=store.embed_dim,
        varied_tags=True, with_hv_payload=False,
    )
    flush_record_buffer(store)
    ids = _read_record_ids(store)
    store.boost_edges(
        [(ids[_ANTI_HIT_RECORD_INDEX], ids[_ANTI_HIT_CONTRADICTING_INDEX])],
        delta=0.1, edge_type="contradicts",
    )
    _build_synthetic_edges(store, _N_RECORDS, _EDGE_FACTOR, _SEED)
    flush_edge_buffer(store)
    return store


_FIXED_CREATED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)
# Frozen Rust age-decay clock: kept >= created_at so the frozen age is 0 by
# equality (the Rust age term also clamps negative age to 0.0). Changing
# this value invalidates every committed byte-parity golden.
_FROZEN_RANK_NOW = int(_FIXED_CREATED_AT.timestamp())
_EXTRA_RECORD_INDEX: dict[str, int] = {"tie_a": 9001, "tie_b": 9002, "lexical_winner": 9003}


def _deterministic_uuid(namespace: int, index: int) -> UUID:
    """A golden bundle must be byte-stable ACROSS separate process runs;
    `make_record`'s own `uuid4()` id and `datetime.now()` timestamp are not.
    Every record built for a golden-hashed fixture gets its id and
    created_at/updated_at overridden to a value derived only from a fixed
    namespace/index, never wall-clock or process entropy."""
    return UUID(int=(namespace << 32) | index)


def _stabilize_record(rec: Any, index: int, *, namespace: int) -> None:
    rec.id = _deterministic_uuid(namespace, index)
    rec.created_at = _FIXED_CREATED_AT
    rec.updated_at = _FIXED_CREATED_AT


def _make_adversarial_extras(dim: int) -> dict[str, Any]:
    """Three hand-built records layered onto the shared corpus:

    - `tie_a`/`tie_b`: identical embedding AND surface -- a genuine scoring
      tie in every candidate feature.
    - `lexical_winner`: literal surface equal to `_ADVERSARIAL_CUE` (a
      guaranteed FTS/substring hit) but embedding is the NEGATION of the
      cue's own embedding (the lowest possible cosine) -- the lexical lane,
      not cosine, must be what pulls it into the candidate set.
    """
    tie_rng = np.random.default_rng(_SEED + 71)
    tie_vec = tie_rng.standard_normal(dim).astype(np.float32)
    tie_vec = (tie_vec / np.linalg.norm(tie_vec)).tolist()
    tie_surface = "alice daily routine note 9001: a tied scoring pair"

    extras_rng = np.random.default_rng(_SEED + 72)
    tie_a = make_record(9001, extras_rng, dim, varied_tags=True, with_hv_payload=False)
    tie_a.embedding = list(tie_vec)
    tie_a.literal_surface = tie_surface
    tie_a.tags = ["bench"]

    tie_b = make_record(9002, extras_rng, dim, varied_tags=True, with_hv_payload=False)
    tie_b.embedding = list(tie_vec)
    tie_b.literal_surface = tie_surface
    tie_b.tags = ["bench"]

    cue_vec = _DeterministicEmbedder(dim=dim).embed(_ADVERSARIAL_CUE)
    lexical_winner = make_record(
        9003, extras_rng, dim, varied_tags=True, with_hv_payload=False,
    )
    lexical_winner.literal_surface = _ADVERSARIAL_CUE
    lexical_winner.embedding = (-np.asarray(cue_vec, dtype=np.float32)).tolist()
    lexical_winner.tags = ["bench", "preference", "fact"]

    for name, rec in (("tie_a", tie_a), ("tie_b", tie_b), ("lexical_winner", lexical_winner)):
        _stabilize_record(rec, _EXTRA_RECORD_INDEX[name], namespace=2)

    return {"tie_a": tie_a, "tie_b": tie_b, "lexical_winner": lexical_winner}


def _seed_adversarial_store(tmp_path) -> tuple[MemoryStore, dict[str, UUID]]:
    """The base fixture reused across this file's spine, trap, and replay
    tests: the shared 400-record corpus plus the three adversarial extras
    above, a tag-boosted record already present via the corpus's own tag
    lattice, and the wired contradicts edge.

    Built record-by-record (not via `build_corpus`) so every id and
    created_at/updated_at can be stabilized -- a committed golden bundle
    must be byte-identical across separate process runs, which `uuid4()`
    and `datetime.now()` can never be.
    """
    store = MemoryStore(path=tmp_path / "adversarial-store")
    dim = store.embed_dim
    corpus_rng = np.random.default_rng(_SEED)
    for i in range(_N_RECORDS):
        rec = make_record(i, corpus_rng, dim, varied_tags=True, with_hv_payload=False)
        _stabilize_record(rec, i, namespace=1)
        store.insert(rec)

    extras = _make_adversarial_extras(dim)
    for rec in extras.values():
        store.insert(rec)
    flush_record_buffer(store)

    ids = _read_record_ids(store)
    store.boost_edges(
        [(ids[_ANTI_HIT_RECORD_INDEX], ids[_ANTI_HIT_CONTRADICTING_INDEX])],
        delta=0.1, edge_type="contradicts",
    )
    _build_synthetic_edges(store, len(ids), _EDGE_FACTOR, _SEED)
    flush_edge_buffer(store)

    named_ids = {name: rec.id for name, rec in extras.items()}
    return store, named_ids


def _raw_bits(arr: "np.ndarray") -> dict[str, Any]:
    """Raw-float-bit encoding of an array: dtype + shape + the exact buffer
    bytes (hex), never a truncated/lossy str() repr."""
    contiguous = np.ascontiguousarray(arr)
    return {
        "__raw_bits__": True,
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "data_hex": contiguous.tobytes().hex(),
    }


def _decode_raw_bits(bits: dict[str, Any]) -> "np.ndarray":
    """Inverse of `_raw_bits`: reconstruct the array for a direct assertion
    (e.g. non-vacuousness) without threading it through the golden path."""
    flat = np.frombuffer(bytes.fromhex(bits["data_hex"]), dtype=bits["dtype"])
    return flat.reshape(bits["shape"])


def _canonicalize(obj: Any) -> Any:
    """Recursively normalize a captured bundle for stable JSON hashing: array
    values become raw-bit dicts, UUID/bytes become strings/hex, dict keys of
    any hashable type become sorted strings. Anything left over falls
    through to `json.dumps`'s own `default=str` on the caller side -- never
    a silent per-type special case here that could mask a future drift."""
    if isinstance(obj, np.ndarray):
        return _raw_bits(obj)
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, dict):
        return {
            str(k): _canonicalize(v)
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(str(v) for v in obj)
    if isinstance(obj, (bytes, bytearray)):
        return obj.hex()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _canonicalize(dataclasses.asdict(obj))
    return obj


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(_canonicalize(obj), sort_keys=True, default=str).encode("utf-8")


def _community_canonical_labels(assignment: "CommunityAssignment") -> dict[UUID, str]:
    """Community ids are freshly minted (`uuid4()`) by every independent
    `detect_communities`/`build_runtime_graph` run -- cross-process noise
    unrelated to content. A golden bundle needs a content-derived, stable
    label instead: the sha256 of the community's own sorted member-id set."""
    members_by_community: dict[UUID, list[str]] = {}
    for member_id, community_id in assignment.node_to_community.items():
        members_by_community.setdefault(community_id, []).append(str(member_id))
    labels: dict[UUID, str] = {}
    for community_id, members in members_by_community.items():
        digest = hashlib.sha256(",".join(sorted(members)).encode("utf-8")).hexdigest()[:16]
        labels[community_id] = f"community:{digest}"
    return labels


def _relabel_communities(obj: Any, labels: dict[UUID, str]) -> Any:
    """Substitute every community-id UUID in a captured bundle with its
    content-derived label from `_community_canonical_labels`, leaving every
    other UUID (record ids) untouched."""
    if isinstance(obj, UUID):
        return labels.get(obj, obj)
    if isinstance(obj, dict):
        return {
            _relabel_communities(k, labels): _relabel_communities(v, labels)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_relabel_communities(v, labels) for v in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _relabel_communities(dataclasses.asdict(obj), labels)
    return obj


def _canonical_response_bytes(response: Any) -> bytes:
    """Deterministic byte serialization: sorted keys so nested dict-key
    ordering never matters, list order preserved (order-significant fields
    like hits/anti_hits stay list-shaped through asdict). Routed through the
    shared raw-bit canonicalizer so an array-valued field, present or future,
    is never silently flattened to a truncated str() repr."""
    return _canonical_json_bytes(dataclasses.asdict(response))


def _bundle_sha256(bundle: dict[str, Any], assignment: "CommunityAssignment") -> str:
    """`response_bytes` is pre-serialized opaque bytes computed before any
    community-id relabeling could reach it -- hashing it here would leak the
    raw (cross-process-random) community id straight back in, undoing the
    relabeling below. `hits`/`anti_hits`/`community_order`/`community_scores`
    already carry the same information in walkable (dict/list/UUID) form, so
    dropping the redundant opaque bytes from the golden hash loses no
    coverage. `rust_scorer_calls` is an engagement-proof counter, not recall
    content -- excluded so it can never perturb the golden.
    """
    labels = _community_canonical_labels(assignment)
    hashable = {
        k: v for k, v in bundle.items()
        if k not in ("response_bytes", "rust_scorer_calls")
    }
    relabeled = _relabel_communities(hashable, labels)
    return hashlib.sha256(_canonical_json_bytes(relabeled)).hexdigest()


def _golden_path(driver: str, fixture_name: str, cue_label: str) -> Path:
    return _GOLDEN_ROOT / driver / f"{fixture_name}__{cue_label}.json"


def _assert_golden(
    driver: str,
    fixture_name: str,
    cue_label: str,
    bundle: dict[str, Any],
    assignment: "CommunityAssignment",
) -> None:
    """Assert the live bundle's sha256 equals the committed per-driver
    golden. Regeneration is opt-in only (`IAI_MCP_RECALL_BYTE_PARITY_REGEN=1`)
    so a silent drift can never auto-rewrite the reference."""
    digest = _bundle_sha256(bundle, assignment)
    path = _golden_path(driver, fixture_name, cue_label)
    if os.environ.get(_GOLDEN_REGEN_ENV) == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"sha256": digest}, indent=2, sort_keys=True) + "\n")
        return
    assert path.exists(), (
        f"missing golden bundle at {path}; regenerate with "
        f"{_GOLDEN_REGEN_ENV}=1 only after confirming the drift is intentional"
    )
    committed = json.loads(path.read_text())["sha256"]
    assert digest == committed, (
        f"byte-parity drift: fixture={fixture_name!r} cue={cue_label!r} "
        f"driver={driver!r} live_sha256={digest} committed_sha256={committed}"
    )


def _serialize_graph_feed(graph: "graph_mod.MemoryGraph") -> bytes:
    """Serialize the graph's full node/edge feed to bytes -- everything a
    fresh `MemoryGraph` needs to reconstruct the identical structure.

    Pickle is safe here: the blob is produced and consumed in-process, in
    the same test, from this test's own synthetic fixture data -- never read
    from an external or untrusted source.
    """
    nodes = []
    for node_id in graph.iter_nodes():
        payload = graph.get_payload(node_id)
        embedding = payload.get("embedding")
        community_id = graph._attrs.get(node_id, {}).get("community_id")
        nodes.append((node_id, community_id, embedding, payload))
    edges = list(graph.iter_edges_with_weight_and_type())
    return pickle.dumps({"nodes": nodes, "edges": edges})


def _reload_graph_from_feed(blob: bytes) -> "graph_mod.MemoryGraph":
    data = pickle.loads(blob)
    fresh = graph_mod.MemoryGraph()
    fresh.clear_and_rebuild(data["nodes"], data["edges"])
    return fresh


def _build_authority_hits(store: MemoryStore, cue_vec: list[float]) -> list[MemoryHit]:
    """Exact-cosine authority candidates, built the same way the production
    `core.dispatch` memory_recall branch builds them (`core/__init__.py`
    around the `_authority_pairs`/`_auth_hits` construction) -- minus the
    live-id-set liveness re-check and mode filter, neither of which a
    hermetic fixture with no concurrent tombstoning can ever exercise."""
    pairs = store.exact_top_k(cue_vec, k=10)
    if not pairs:
        return []
    ids = [rid for rid, _score in pairs]
    recs = store.get_batch(ids)
    hits: list[MemoryHit] = []
    for rid, score in pairs:
        rec = recs.get(rid)
        if rec is None:
            continue
        hits.append(MemoryHit(
            record_id=rid,
            score=float(score),
            reason="exact-cosine",
            literal_surface=rec.literal_surface or "",
            adjacent_suggestions=[],
            session_id=(rec.provenance or [{}])[0].get("session_id"),
            captured_at=rec.created_at.isoformat() if rec.created_at else None,
            community_id=getattr(rec, "community_id", None),
            epistemic_status=rec.epistemic_status,
            salience_level=rec.salience_level,
        ))
    return hits


def _capture_recall(
    *,
    store: MemoryStore,
    graph,
    assignment,
    rich_club: list[UUID],
    embedder: _DeterministicEmbedder,
    cue: str,
    budget_tokens: int = 1500,
) -> dict[str, Any]:
    """Run one recall with every capture point wrapped; return the captures.

    Every monkeypatch is scoped to this single call via
    `pytest.MonkeyPatch.context()` -- the frozen clock (and every spy) is
    gone the moment this function returns, so it can never leak into a
    caller's own patching or a later capture in the same test.
    """
    captured: dict[str, Any] = {}

    orig_pool = pipeline_mod._collect_graph_pool
    orig_seeds = pipeline_mod._pick_seeds
    orig_gate = pipeline_mod._community_gate_max_node_scored
    orig_normalize = pipeline_mod._normalize_pool
    orig_flags = pipeline_mod._t11_t12_flags
    orig_two_hop = graph_mod.MemoryGraph.two_hop_neighborhood_with_provenance

    def _spy_pool(*a, **kw):
        result = orig_pool(*a, **kw)
        captured["pool_ids"] = list(result[0])
        return result

    def _spy_seeds(*a, **kw):
        result = orig_seeds(*a, **kw)
        captured["seed_indices"] = [int(i) for i in result]
        return result

    def _spy_gate(*a, **kw):
        result = orig_gate(*a, **kw)
        captured["community_order"] = list(result[0])
        captured["community_scores"] = dict(result[1])
        return result

    def _spy_normalize(*a, **kw):
        result = orig_normalize(*a, **kw)
        captured["normalized_pool"] = np.array(result, dtype=np.float32, copy=True)
        return result

    def _spy_flags(*a, **kw):
        result = orig_flags(*a, **kw)
        captured["t11_flags"] = np.array(result[0], dtype=bool, copy=True)
        captured["t12_flags"] = np.array(result[1], dtype=bool, copy=True)
        return result

    def _spy_two_hop(self, seeds, top_k=5):
        result = orig_two_hop(self, seeds, top_k=top_k)
        captured["provenance"] = dict(result)
        return result

    def _spy_rank_now() -> int:
        captured["rank_now_calls"] = captured.get("rank_now_calls", 0) + 1
        return _FROZEN_RANK_NOW

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pipeline_mod, "_collect_graph_pool", _spy_pool)
        mp.setattr(pipeline_mod, "_pick_seeds", _spy_seeds)
        mp.setattr(pipeline_mod, "_community_gate_max_node_scored", _spy_gate)
        mp.setattr(pipeline_mod, "_normalize_pool", _spy_normalize)
        mp.setattr(pipeline_mod, "_t11_t12_flags", _spy_flags)
        mp.setattr(
            graph_mod.MemoryGraph, "two_hop_neighborhood_with_provenance", _spy_two_hop,
        )
        # Freeze both clocks the Rust-scorer path reads: `_age_penalty`
        # (richclub.py/session.py, function-local imports) and `_rank_now`
        # (the Rust `.score()` age term). Both frozen to the same instant as
        # every seeded record's created_at, so age == 0 on both lanes.
        mp.setattr(pipeline_mod, "_age_penalty", lambda created_at: 0.0)
        mp.setattr(pipeline_mod, "_rank_now", _spy_rank_now)

        response = recall_for_response(
            store=store, graph=graph, assignment=assignment, rich_club=rich_club,
            embedder=embedder, cue=cue, session_id="recall-byte-parity-gate",
            budget_tokens=budget_tokens,
            # Exercise the live-served hybrid-Rust path -- `core.dispatch`
            # always requests it. No `structural_weight` override: its
            # default (0.0) is what lets the Rust path engage.
            use_rust_scorer=True,
        )

        cue_vec_list = embedder.embed(cue)
        cue_vec = pipeline_mod._sanitize_vec(np.asarray(cue_vec_list, dtype=np.float32))
        cue_norm = float(np.linalg.norm(cue_vec))
        if cue_norm > 0.0:
            cue_vec = cue_vec / cue_norm

        normalized_pool = captured.get("normalized_pool")
        if normalized_pool is not None and normalized_pool.size:
            recomputed_cosine = np.matmul(normalized_pool, cue_vec).astype(np.float32)
        else:
            recomputed_cosine = np.empty(0, dtype=np.float32)

        authority_hits = _build_authority_hits(store, cue_vec_list)
        merged_hits, merged_budget = merge_authority_hits(
            list(response.hits), authority_hits, budget_tokens,
        )
        response.hits = merged_hits
        response.budget_used = merged_budget

    pool_ids = captured.get("pool_ids", [])
    seed_ids = [pool_ids[i] for i in captured.get("seed_indices", [])]
    t11_flags = captured.get("t11_flags", np.zeros(0, dtype=bool))
    t12_flags = captured.get("t12_flags", np.zeros(0, dtype=bool))
    empty_pool = np.zeros((0, 0), dtype=np.float32)

    return {
        "community_order": captured.get("community_order", []),
        "community_scores": captured.get("community_scores", {}),
        "seed_ids": seed_ids,
        "provenance": captured.get("provenance", {}),
        "hits": list(response.hits),
        "anti_hits": list(response.anti_hits),
        "response_bytes": _canonical_response_bytes(response),
        "normalized_pool_bits": _raw_bits(captured.get("normalized_pool", empty_pool)),
        "cosine_bits": _raw_bits(recomputed_cosine),
        "t11_flags_bits": _raw_bits(t11_flags),
        "t12_flags_bits": _raw_bits(t12_flags),
        "merged_scores_reasons": [
            (str(h.record_id), h.score, h.reason) for h in merged_hits
        ],
        # Proof the Rust scorer path engaged, not the sha256-hashed bundle
        # (excluded in `_bundle_sha256`) -- a call count is an
        # implementation detail, not recall content.
        "rust_scorer_calls": captured.get("rank_now_calls", 0),
    }


_COMPARABLE_BUNDLE_KEYS: tuple[str, ...] = (
    "community_order",
    "community_scores",
    "seed_ids",
    "provenance",
    "hits",
    "anti_hits",
    "response_bytes",
    "normalized_pool_bits",
    "cosine_bits",
    "t11_flags_bits",
    "t12_flags_bits",
    "merged_scores_reasons",
)


@pytest.fixture(params=["stdlib", "lilli"])
def storage_driver(request, monkeypatch: pytest.MonkeyPatch) -> str:
    if request.param == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built -- lilli driver unavailable")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    return request.param


def test_capture_points_deterministic_and_non_vacuous(
    tmp_path, monkeypatch, storage_driver,
):
    """Old-vs-old on current code: every capture point is deterministic, and
    the multi-community cue produces a non-empty capture at every point.
    Parametrized over both storage drivers -- the corpus/graph/recall seam
    is driver-visible (MemoryStore reads LILLI_STORAGE_DRIVER)."""
    store = _seed_store(tmp_path)
    graph, assignment, rich_club = retrieve.build_runtime_graph(store)
    embedder = _DeterministicEmbedder(dim=store.embed_dim)

    for label, cue in _CUES.items():
        run1 = _capture_recall(
            store=store, graph=graph, assignment=assignment,
            rich_club=rich_club, embedder=embedder, cue=cue,
        )
        run2 = _capture_recall(
            store=store, graph=graph, assignment=assignment,
            rich_club=rich_club, embedder=embedder, cue=cue,
        )

        assert run1["community_scores"] == run2["community_scores"], (
            f"[{label}] community_scores drifted across identical runs"
        )
        assert run1["community_order"] == run2["community_order"], (
            f"[{label}] community_order drifted across identical runs"
        )
        assert run1["seed_ids"] == run2["seed_ids"], (
            f"[{label}] seed_ids drifted across identical runs"
        )
        assert run1["provenance"] == run2["provenance"], (
            f"[{label}] provenance drifted across identical runs"
        )
        assert run1["hits"] == run2["hits"], (
            f"[{label}] hits drifted across identical runs"
        )
        assert run1["anti_hits"] == run2["anti_hits"], (
            f"[{label}] anti_hits drifted across identical runs"
        )
        assert run1["response_bytes"] == run2["response_bytes"], (
            f"[{label}] final response bytes drifted across identical runs"
        )

        if label == "multi_community":
            assert run1["community_scores"], "community_scores must be non-empty"
            assert len(set(run1["community_order"])) >= 1
            assert run1["seed_ids"], "seed_ids must be non-empty"
            assert run1["provenance"], "provenance map must be non-empty"
            assert run1["hits"], "hits must be non-empty"

        if label == "anti_hit_expected":
            assert run1["anti_hits"], (
                "the wired contradicts-edge record did not surface an "
                "anti-hit for its own exact-surface cue"
            )


def _any_artifact_differs(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return any(a[key] != b[key] for key in _COMPARABLE_BUNDLE_KEYS)


def _perturb_one_char(text: str) -> str:
    """Flip the first alphabetic character to a distinct letter."""
    for i, ch in enumerate(text):
        if ch.isalpha():
            flipped = "z" if ch.lower() != "z" else "a"
            if ch.isupper():
                flipped = flipped.upper()
            return text[:i] + flipped + text[i + 1:]
    return text + "x"


def test_negative_control_sensitivity_not_vacuous(tmp_path, monkeypatch, storage_driver):
    """The byte-parity comparison must be SENSITIVE, not vacuous: perturbing
    one cue character, or one record's embedding, must change at least one
    captured artifact, and the unperturbed control must still match itself
    before and after."""
    store = _seed_store(tmp_path)
    graph, assignment, rich_club = retrieve.build_runtime_graph(store)
    embedder = _DeterministicEmbedder(dim=store.embed_dim)
    base_cue = _CUES["multi_community"]

    control_a = _capture_recall(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder, cue=base_cue,
    )
    control_b = _capture_recall(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder, cue=base_cue,
    )
    assert control_a["response_bytes"] == control_b["response_bytes"], (
        "control run must match itself before any perturbation"
    )

    # (a) perturb one cue character.
    perturbed_cue = _perturb_one_char(base_cue)
    assert perturbed_cue != base_cue
    cue_perturbed = _capture_recall(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder, cue=perturbed_cue,
    )
    assert _any_artifact_differs(control_a, cue_perturbed), (
        "flipping one cue character produced IDENTICAL captures -- "
        "the byte-parity gate would be vacuous"
    )

    # (b) perturb one record's embedding directly on the already-built graph
    # (no store round-trip needed -- `_collect_graph_pool` reads
    # `graph.get_embedding`), then revert it in a `finally` so the control
    # re-check below runs against the original, unperturbed graph. The
    # target is one of the control run's OWN seed ids (guaranteed to sit
    # inside the reachable candidate set for this cue) -- an arbitrary node
    # outside the gated communities/candidate pool can perturb invisibly.
    assert control_a["seed_ids"], "control run must have at least one seed to perturb"
    target_id = control_a["seed_ids"][0]
    original_embedding = graph.get_embedding(target_id).copy()
    nudged = -original_embedding
    graph.set_node_payload(target_id, {"embedding": nudged.tolist()})
    try:
        emb_perturbed = _capture_recall(
            store=store, graph=graph, assignment=assignment,
            rich_club=rich_club, embedder=embedder, cue=base_cue,
        )
    finally:
        graph.set_node_payload(target_id, {"embedding": original_embedding.tolist()})

    assert _any_artifact_differs(control_a, emb_perturbed), (
        "nudging one record's embedding produced IDENTICAL captures -- "
        "the byte-parity gate would be vacuous"
    )

    # Control still holds after the embedding perturbation was reverted.
    control_c = _capture_recall(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder, cue=base_cue,
    )
    assert control_a["response_bytes"] == control_c["response_bytes"], (
        "control run drifted after the embedding perturbation was reverted"
    )


def test_spine_deep_intermediates_and_authority_inclusion_are_captured(
    tmp_path, storage_driver,
):
    """The adversarial base fixture wired end-to-end: deep intermediates are
    non-vacuous raw-float-bit captures, the hand-built tie pair ties in the
    normalized pool, and a small token budget forces truncation the token
    budget-pack must produce byte-stable results for.

    The capture runs the live hybrid-Rust scorer path (`use_rust_scorer=True`,
    both clocks frozen), so T11/T12 are populated here, proven non-vacuous by
    dtype/shape below. This cue is not identifier-grade, so the warm-BM25
    lexical lane stays disabled; the low-cosine lexical winner is already
    reachable via the full cosine argsort like every other pool candidate --
    FTS/T11/T12 only multiply an already-admitted candidate's score, never
    widen the pool.
    """
    store, adversarial_ids = _seed_adversarial_store(tmp_path)
    graph, assignment, rich_club = retrieve.build_runtime_graph(store)
    embedder = _DeterministicEmbedder(dim=store.embed_dim)

    bundle = _capture_recall(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_ADVERSARIAL_CUE, budget_tokens=1500,
    )

    assert bundle["normalized_pool_bits"]["shape"][0] > 0, "candidate pool must be non-empty"
    assert bundle["normalized_pool_bits"]["dtype"] == "float32"
    assert bundle["cosine_bits"]["shape"][0] == bundle["normalized_pool_bits"]["shape"][0]
    assert bundle["t11_flags_bits"]["dtype"] == "bool"
    assert bundle["t12_flags_bits"]["dtype"] == "bool"
    assert bundle["merged_scores_reasons"], "authority-merged candidate list must be non-empty"
    assert bundle["rust_scorer_calls"] >= 1, (
        "the Rust scorer path did not engage -- frozen _rank_now never called"
    )

    tie_a_id, tie_b_id = adversarial_ids["tie_a"], adversarial_ids["tie_b"]
    tie_a_emb = graph.get_embedding(tie_a_id)
    tie_b_emb = graph.get_embedding(tie_b_id)
    assert tie_a_emb is not None and tie_b_emb is not None
    assert np.array_equal(tie_a_emb, tie_b_emb), (
        "the hand-built tie pair must carry byte-identical embeddings"
    )

    small_budget_bundle = _capture_recall(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_ADVERSARIAL_CUE, budget_tokens=20,
    )
    assert len(small_budget_bundle["hits"]) < len(bundle["hits"]), (
        "a much smaller token budget must truncate the packed hit list -- "
        "the budget boundary fixture condition would otherwise be vacuous"
    )


def test_existing_vs_snapshot_differential_is_byte_identical(tmp_path, storage_driver):
    """Serialize the live graph's feed to bytes, reload it into a fresh
    `MemoryGraph`, and assert the reloaded reader reproduces the live
    reader's full bundle byte-for-byte. Today this is an IDENTITY round-trip
    (no alternate feed representation exists yet) -- a later lever that
    introduces one must keep this green unchanged."""
    store, _adversarial_ids = _seed_adversarial_store(tmp_path)
    graph, assignment, rich_club = retrieve.build_runtime_graph(store)
    embedder = _DeterministicEmbedder(dim=store.embed_dim)

    live_bundle = _capture_recall(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_ADVERSARIAL_CUE, budget_tokens=1500,
    )

    feed_bytes = _serialize_graph_feed(graph)
    reloaded_graph = _reload_graph_from_feed(feed_bytes)

    reloaded_bundle = _capture_recall(
        store=store, graph=reloaded_graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_ADVERSARIAL_CUE, budget_tokens=1500,
    )

    for key in _COMPARABLE_BUNDLE_KEYS:
        assert live_bundle[key] == reloaded_bundle[key], (
            f"snapshot/alternate-fed reader diverged from the live reader "
            f"on bundle field {key!r}"
        )


def test_golden_bundle_matches_committed_per_driver(tmp_path, storage_driver):
    """The base fixture's bundle hash equals the committed per-driver golden
    -- the pre-condition gate every later allocation-reducing lever must keep
    green before it ships."""
    store, _adversarial_ids = _seed_adversarial_store(tmp_path)
    graph, assignment, rich_club = retrieve.build_runtime_graph(store)
    embedder = _DeterministicEmbedder(dim=store.embed_dim)

    bundle = _capture_recall(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_ADVERSARIAL_CUE, budget_tokens=1500,
    )

    assert bundle["rust_scorer_calls"] >= 1, (
        "the Rust scorer path did not engage -- frozen _rank_now never called"
    )
    _assert_golden(storage_driver, "adversarial_base", "adversarial_cue", bundle, assignment)


# ---------------------------------------------------------------------------
# Adversarial trap fixtures: each isolates one resident rank-index feed
# source or scoring input and proves it both load-bearing (golden) and
# non-vacuous (a targeted perturbation changes the captured bundle).
# ---------------------------------------------------------------------------

_DRIFT_TOKEN = "drift_scan_anchor_token"
_PENDING_TOKEN = "pending_second_source_token"
_HOOK_CUE = "resident_hook_candidate"
_CENTRALITY_CUE = "boundary seed selection probe phrase for centrality"
_AUTHORITY_PUBLISH_CUE = "authority merge publish race boundary probe cue"


def _toggle_hook_insert(store: MemoryStore, record: Any) -> None:
    """Insert `record` into the store's DB with the graph-sync hook
    temporarily disabled, so it lands in the SQL table without ever
    touching the graph or the resident rank-index handle -- the
    store/graph divergence the drift and pending traps depend on.
    `register_graph_sync_hook` is a single-slot overwrite (`_store.py`), so
    saving and restoring it is safe even when a hook was already wired by
    `retrieve.build_runtime_graph`.
    """
    saved_hook = store._graph_sync_hook
    store.register_graph_sync_hook(None)
    try:
        store.insert(record)
        flush_record_buffer(store)
    finally:
        store.register_graph_sync_hook(saved_hook)


def _reset_rank_index(store: MemoryStore) -> None:
    """Force the next `.score()`/`.snapshot()` call to cold-build the
    resident rank index from current store state. `.score()` only builds
    when `_index is None` -- it never rebuilds on its own for a store
    content change or even a graph-identity mismatch (`_RankIndexHandle`
    docstrings). Deleting the store-attached handle attribute makes
    `rank_index_for` mint a fresh one on next access."""
    from iai_mcp.store._rank_index import _HANDLE_ATTR

    if hasattr(store, _HANDLE_ATTR):
        delattr(store, _HANDLE_ATTR)


def _score_for(bundle: dict[str, Any], record_id: UUID) -> "float | None":
    target = str(record_id)
    for rid, score, _reason in bundle["merged_scores_reasons"]:
        if rid == target:
            return score
    return None


def test_trap_drift_under_cache_hit_lag(tmp_path, monkeypatch, storage_driver):
    """Trap A: a reachable probe record's captured Rust partial_score
    reflects the corpus-wide BM25 stats the resident rank index scans
    directly off the store (`_rank_index.py:372`) -- including a drift
    record present in the store but absent from the graph pool. The graph
    is built BEFORE the drift record is written (with the graph-sync hook
    disabled for that one write), reproducing the store>graph divergence a
    cache-hit path serves at runtime; the rank index's own first build
    happens later still (triggered by this test's first `_capture_recall`),
    so it sees the store WITH the drift record already present. This trap
    guards the rank-index feed SOURCE (whole-store scan), not any
    particular cache path.

    Pins `IAI_MCP_LEX_MIN_IDF=0.0` to isolate the BM25 membership
    mechanism under test; this does not gate the production IDF default
    (4.0).
    """
    monkeypatch.delenv("IAI_MCP_LEX_FUSION_OFF", raising=False)
    monkeypatch.setenv("IAI_MCP_LEX_MIN_IDF", "0.0")

    store, _adversarial_ids = _seed_adversarial_store(tmp_path)
    dim = store.embed_dim
    embedder = _DeterministicEmbedder(dim=dim)

    probe_rng = np.random.default_rng(_SEED + 91)
    probe = make_record(9101, probe_rng, dim, varied_tags=True, with_hv_payload=False)
    probe.embedding = list(embedder.embed(_DRIFT_TOKEN))
    probe.literal_surface = f"{_DRIFT_TOKEN} carried by the reachable probe record"
    probe.tags = ["bench"]
    _stabilize_record(probe, 0, namespace=3)
    store.insert(probe)
    flush_record_buffer(store)
    assert probe.id == _deterministic_uuid(3, 0), (
        "pattern separation must never rewrite the probe's stabilized id"
    )

    graph, assignment, rich_club = retrieve.build_runtime_graph(store)

    drift_rng = np.random.default_rng(_SEED + 92)
    drift = make_record(9102, drift_rng, dim, varied_tags=True, with_hv_payload=False)
    drift.literal_surface = f"{_DRIFT_TOKEN} also present on the drift-only record"
    drift.tags = ["bench"]
    _stabilize_record(drift, 0, namespace=4)
    _toggle_hook_insert(store, drift)
    assert drift.id == _deterministic_uuid(4, 0), (
        "pattern separation must never rewrite the drift record's stabilized id"
    )
    assert str(drift.id) not in graph._node_payload, (
        "the drift record must never reach the graph payload -- it is a "
        "store-only row by construction"
    )

    assert pipeline_mod._cue_identifier_grade(_DRIFT_TOKEN) is True

    bundle_with_drift = _capture_recall(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_DRIFT_TOKEN, budget_tokens=1500,
    )
    assert bundle_with_drift["rust_scorer_calls"] >= 1

    score_with_drift = _score_for(bundle_with_drift, probe.id)
    assert score_with_drift is not None, "the reachable probe record must surface as a hit"

    _assert_golden(
        storage_driver, "trap_drift_active_scan", "drift_token",
        bundle_with_drift, assignment,
    )

    # Sensitivity: delete the drift record and force a cold rank-index
    # rebuild (`.score()` never rebuilds on its own -- see `_reset_rank_index`).
    store.delete(drift.id)
    _reset_rank_index(store)
    bundle_without_drift = _capture_recall(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_DRIFT_TOKEN, budget_tokens=1500,
    )
    score_without_drift = _score_for(bundle_without_drift, probe.id)
    assert score_without_drift is not None
    assert score_with_drift != score_without_drift, (
        "deleting the drift record must change the reachable probe's "
        "partial_score -- otherwise the BM25 active-scan drift source is "
        "vacuous"
    )
    assert _any_artifact_differs(bundle_with_drift, bundle_without_drift), (
        "deleting the drift record must change at least one captured "
        "artifact -- otherwise the trap is vacuous"
    )


def test_trap_pending_second_source_membership(tmp_path, monkeypatch, storage_driver):
    """Trap C: a pending record (`embedding_pending=1`), excluded from the
    graph pool the same way the drift record is, moves a reachable probe's
    captured partial_score exactly as Trap A does -- but via the SEPARATE
    `_bulk_pending_records` second source (`_rank_index.py:426`), not the
    active-scan path Trap A exercises. `store.insert_pending` writes
    directly at the DB layer and never fires the graph-sync hook, so no
    hook toggling is needed here (unlike Trap A's active-record insert).

    The pending record never surfaces as a hit (fact: `_collect_graph_pool`
    excludes pending rows at the SQL level, so it can never gain an
    `id_to_idx` entry) -- its `tier`/`created_at` fields are therefore never
    observable through this harness. Only its BM25 membership effect on a
    reachable record is checkable, and that is what this trap asserts.

    Pins `IAI_MCP_LEX_MIN_IDF=0.0` to isolate the BM25 membership
    mechanism under test; this does not gate the production IDF default
    (4.0).
    """
    monkeypatch.delenv("IAI_MCP_LEX_FUSION_OFF", raising=False)
    monkeypatch.setenv("IAI_MCP_LEX_MIN_IDF", "0.0")

    store, _adversarial_ids = _seed_adversarial_store(tmp_path)
    dim = store.embed_dim
    embedder = _DeterministicEmbedder(dim=dim)

    probe_rng = np.random.default_rng(_SEED + 93)
    probe = make_record(9103, probe_rng, dim, varied_tags=True, with_hv_payload=False)
    probe.embedding = list(embedder.embed(_PENDING_TOKEN))
    probe.literal_surface = f"{_PENDING_TOKEN} carried by the reachable probe record"
    probe.tags = ["bench"]
    _stabilize_record(probe, 0, namespace=3)
    store.insert(probe)
    flush_record_buffer(store)
    assert probe.id == _deterministic_uuid(3, 0), (
        "pattern separation must never rewrite the probe's stabilized id"
    )

    graph, assignment, rich_club = retrieve.build_runtime_graph(store)

    pending_id = _deterministic_uuid(5, 0)
    fixed_ts = _FIXED_CREATED_AT.isoformat()
    store.insert_pending(
        record_id=str(pending_id),
        tier="episodic",
        literal_surface=f"{_PENDING_TOKEN} also present on the pending-only record",
        tags_json=json.dumps(["bench"]),
        provenance_json="[]",
        created_at=fixed_ts,
        updated_at=fixed_ts,
    )
    assert str(pending_id) not in graph._node_payload, (
        "a pending record must never reach the graph payload"
    )

    assert pipeline_mod._cue_identifier_grade(_PENDING_TOKEN) is True

    bundle_with_pending = _capture_recall(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_PENDING_TOKEN, budget_tokens=1500,
    )
    assert bundle_with_pending["rust_scorer_calls"] >= 1

    score_with_pending = _score_for(bundle_with_pending, probe.id)
    assert score_with_pending is not None, "the reachable probe record must surface as a hit"
    assert str(pending_id) not in [
        rid for rid, _s, _r in bundle_with_pending["merged_scores_reasons"]
    ], "the pending record itself must never surface as a hit"

    _assert_golden(
        storage_driver, "trap_pending_second_source", "pending_token",
        bundle_with_pending, assignment,
    )

    # Sensitivity: delete the pending record and force a cold rank-index
    # rebuild -- identical mechanism to Trap A's sensitivity, distinct
    # feed source (the pending second source, not the active scan).
    store.delete(pending_id)
    _reset_rank_index(store)
    bundle_without_pending = _capture_recall(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_PENDING_TOKEN, budget_tokens=1500,
    )
    score_without_pending = _score_for(bundle_without_pending, probe.id)
    assert score_without_pending is not None
    assert score_with_pending != score_without_pending, (
        "deleting the pending record must change the reachable probe's "
        "partial_score -- otherwise the pending second-source membership "
        "path is vacuous"
    )
    assert _any_artifact_differs(bundle_with_pending, bundle_without_pending), (
        "deleting the pending record must change at least one captured "
        "artifact -- otherwise the trap is vacuous"
    )


def test_trap_hook_mutation_no_rebuild_and_deterministic(
    tmp_path, monkeypatch, storage_driver,
):
    """Trap D: a write-time hook op is drained through the resident rank
    index's own incremental `feed()`, never a Python-side `_build()`
    rebuild (`_RankIndexHandle._needs_build`/`feed` docstrings). Asserts
    the REAL invariants -- there is no
    stamp-before/recheck-after/retry discipline to guard instead:

      (i)   the hook-fed insert never triggers a fresh `_build()` -- proven
            by object identity on `handle._index` across the op, not by any
            generation/staleness flag.
      (ii)  two `_capture_recall`s at an unchanged state are byte-identical,
            both before the hook op and again after it is drained.
      (iii) ordering is load-bearing, and stronger than `feed()`-before-
            first-build alone: `feed()` QUEUES a write into the resident
            index; a plain `.score()` call (what `recall_for_response` /
            `_capture_recall` runs) does NOT drain that queue on its own --
            `overlay_len()` stays at its pre-hook value and the hook-fed
            record does not yet surface among the Rust-scored hits in a
            recall taken right after the hook. Only an explicit
            `.snapshot()` call drains it (confirmed
            empirically: `overlay_len()` goes from the pre-hook value to
            pre-hook+1 only after `.snapshot()`, never right after `feed()`
            alone). In production this drain is `core/__init__.py`'s
            `_rank_handle.snapshot()`, called BEFORE the
            pipeline's own `.score()` -- a step outside `recall_for_response`
            that this harness must therefore call explicitly to reproduce
            the real, visible-after-write behavior.

    Sensitivity is proven separately: a hook op that mutates an EXISTING
    reachable pool member's embedding, drained the same way, must change
    the captured bundle.

    Pins `IAI_MCP_LEX_MIN_IDF=0.0` to isolate the mechanism under test;
    this does not gate the production IDF default (4.0).
    """
    monkeypatch.delenv("IAI_MCP_LEX_FUSION_OFF", raising=False)
    monkeypatch.setenv("IAI_MCP_LEX_MIN_IDF", "0.0")

    store, _adversarial_ids = _seed_adversarial_store(tmp_path)
    dim = store.embed_dim
    embedder = _DeterministicEmbedder(dim=dim)
    graph, assignment, rich_club = retrieve.build_runtime_graph(store)

    from iai_mcp.store._rank_index import rank_index_for

    handle = rank_index_for(store)
    assert handle._index is None, "the resident rank index must not be built yet"

    # (iii) ordering: the first `.score()` call (inside `_capture_recall`)
    # is what builds the index for the first time.
    bundle_pre_1 = _capture_recall(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_HOOK_CUE, budget_tokens=1500,
    )
    assert bundle_pre_1["rust_scorer_calls"] >= 1
    index_after_first_build = handle._index
    assert index_after_first_build is not None

    # (ii) repeat-determinism BEFORE the hook op.
    bundle_pre_2 = _capture_recall(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_HOOK_CUE, budget_tokens=1500,
    )
    for key in _COMPARABLE_BUNDLE_KEYS:
        assert bundle_pre_1[key] == bundle_pre_2[key], (
            f"repeat capture at an unchanged state diverged on {key!r} "
            "before any hook op"
        )
    assert handle._index is index_after_first_build, (
        "a repeat capture at unchanged state must never trigger a rebuild"
    )

    # (i) drive one hook op IN-PROCESS: an insert that becomes a real graph
    # pool member. `_make_graph_sync_hook` also feeds the resident rank
    # index (store passed explicitly), queuing the write via `feed()`,
    # never a Python-side rebuild.
    hook = retrieve._make_graph_sync_hook(graph, store)
    new_rng = np.random.default_rng(_SEED + 94)
    new_record = make_record(9104, new_rng, dim, varied_tags=True, with_hv_payload=False)
    new_record.embedding = list(embedder.embed(_HOOK_CUE))
    new_record.literal_surface = f"{_HOOK_CUE} fed in-process after the first build"
    new_record.tags = ["bench"]
    _stabilize_record(new_record, 0, namespace=6)
    overlay_len_before = handle.overlay_len()
    hook("insert", new_record)
    assert str(new_record.id) in graph._node_payload, (
        "the hook-fed insert must land in the graph payload"
    )
    assert handle._index is index_after_first_build, (
        "a hook-fed insert on the already-bound graph must never trigger a "
        "Python-side rebuild"
    )
    assert handle.overlay_len() == overlay_len_before, (
        "feed() alone must not yet be visible in overlay_len() -- it "
        "becomes visible only once .snapshot() drains it (iii)"
    )

    # (iii) a recall RIGHT AFTER the hook, with no explicit drain: the
    # RUST-SCORED fields (`merged_scores_reasons`/`hits`/`response_bytes`,
    # sourced from `.score()`'s `winners`) must NOT yet reflect the fed
    # record -- `.score()` never drains the queue on its own. Pool/seed
    # selection is a SEPARATE, pure-Python read straight off the graph and
    # updates immediately regardless of the Rust drain state -- so this
    # trap asserts the narrower, empirically-true claim on the Rust-scored
    # slice only, not the whole bundle.
    bundle_undrained = _capture_recall(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_HOOK_CUE, budget_tokens=1500,
    )
    assert str(new_record.id) not in [
        rid for rid, _s, _r in bundle_undrained["merged_scores_reasons"]
    ], (
        "the hook-fed record must NOT yet surface as a scored hit before "
        "the drain -- .score() reads only the committed Rust structures, "
        "never the undrained feed() queue"
    )
    assert handle._index is index_after_first_build, (
        "an undrained recall must still never trigger a rebuild"
    )

    # Drain explicitly, mirroring core/__init__.py's snapshot-before-
    # score sequence -- the step this harness's `_capture_recall` never
    # performs on its own.
    handle.snapshot()
    assert handle._index is index_after_first_build, (
        "draining via .snapshot() must never trigger a Python-side rebuild "
        "either -- it only drains the Rust engine's own delta overlay"
    )
    assert handle.overlay_len() > overlay_len_before, (
        "the hook-fed insert must actually reach the resident rank index's "
        "overlay once drained -- proving _make_graph_sync_hook's own "
        "rank-index-feed except-Exception clause did not silently swallow "
        "the write"
    )

    bundle_post_1 = _capture_recall(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_HOOK_CUE, budget_tokens=1500,
    )
    assert handle._index is index_after_first_build, (
        "a recall after the drain must still never trigger a rebuild"
    )
    assert str(new_record.id) in [
        rid for rid, _s, _r in bundle_post_1["merged_scores_reasons"]
    ], (
        "the hook-fed record must surface as a scored hit once drained -- "
        "otherwise the feed()-then-snapshot() path is vacuous"
    )

    # (ii) repeat-determinism AFTER the drain, at the new state.
    bundle_post_2 = _capture_recall(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_HOOK_CUE, budget_tokens=1500,
    )
    for key in _COMPARABLE_BUNDLE_KEYS:
        assert bundle_post_1[key] == bundle_post_2[key], (
            f"repeat capture at an unchanged post-drain state diverged on "
            f"{key!r}"
        )

    _assert_golden(
        storage_driver, "trap_hook_mutation", "hook_cue", bundle_post_1, assignment,
    )

    # Sensitivity: a hook op that mutates an EXISTING reachable pool
    # member's embedding, drained the same way, must change the bundle.
    mutate_target = new_record
    mutate_target.embedding = list(-np.asarray(mutate_target.embedding, dtype=np.float32))
    hook("update", mutate_target)
    handle.snapshot()
    assert handle._index is index_after_first_build, (
        "an update hook op (plus its drain) must also never trigger a rebuild"
    )
    bundle_mutated = _capture_recall(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_HOOK_CUE, budget_tokens=1500,
    )
    assert _any_artifact_differs(bundle_post_1, bundle_mutated), (
        "mutating a reachable pool member's embedding via the hook must "
        "change at least one captured artifact -- otherwise the hook-feed "
        "path is vacuous"
    )


def _find_seed_boundary_node(graph, store, embedder, cue: str) -> UUID:
    """Locate a genuine seed-selection boundary node for `cue`: the pool
    candidate ranked 4th by raw cosine (the position immediately outside
    `_pick_seeds`'s top-3 blend under the corpus's natural, near-zero
    centrality distribution). Approximates the seed blend over the ungated
    pool (`_collect_graph_pool` + `_normalize_pool` + `_sanitize_vec` +
    `_pick_seeds`) read-only, over the SAME graph the trap's captures run
    against -- it reads `graph.get_centrality` directly and does not
    replicate the pipeline's `centrality_for_runtime` recompute-once path
    (which overwrites `centrality_arr` when `not graph._centrality_resolved`).
    The boundary this helper returns is confirmed correct by the trap's own
    baseline/positive assertions below, not by this helper alone."""
    cue_vec_list = embedder.embed(cue)
    cue_vec = pipeline_mod._sanitize_vec(np.asarray(cue_vec_list, dtype=np.float32))
    norm = float(np.linalg.norm(cue_vec))
    if norm > 0.0:
        cue_vec = cue_vec / norm

    pool_ids, pool_embs = pipeline_mod._collect_graph_pool(graph, None, store)
    pool_embs = pipeline_mod._normalize_pool(graph, pool_ids, pool_embs)
    shared_cos = np.matmul(pool_embs, cue_vec).astype(np.float32)
    cosine_top_indices = np.argsort(-shared_cos, kind="stable")

    centrality_arr = np.zeros(len(pool_ids), dtype=np.float32)
    for i, rid in enumerate(pool_ids):
        centrality_arr[i] = float(graph.get_centrality(rid))

    seeds_before = pipeline_mod._pick_seeds(cosine_top_indices, shared_cos, centrality_arr, n=3)
    boundary_idx = int(cosine_top_indices[3])
    assert boundary_idx not in [int(i) for i in seeds_before], (
        "the rank-4-by-cosine candidate is already a seed under the "
        "corpus's natural centrality distribution -- pick a different "
        "boundary construction"
    )
    return pool_ids[boundary_idx]


def test_trap_centrality_source_is_graph_payload(tmp_path, monkeypatch, storage_driver):
    """Trap B (corrected attribution): recall's seed centrality is the GRAPH
    PAYLOAD value read via `graph.get_centrality` -> `centrality_arr` ->
    `_pick_seeds` -> `seed_ids` -- never the rank-index resident/store
    `centrality` column, which `fused_score` never reads (verified against
    `lib.rs`'s scoring loop: `Inner.centrality` is stored on the struct but
    not consulted anywhere in `fused_score`'s body).

    Two-sided sensitivity over the SAME boundary node T:
      (positive) mutating T's GRAPH PAYLOAD centrality across the seed
      boundary changes `seed_ids` (and the golden).
      (negative) mutating T's STORE COLUMN centrality only, with the graph
      payload left untouched, THEN forcing a cold rank-index rebuild so the
      write is actually pulled into the Rust engine's `Inner.centrality`
      field (`_rank_index.py` selects the store's `centrality` column on a
      cold build), changes NOTHING -- proving `fused_score` itself ignores
      that field. The forced rebuild is what makes this a real discriminant:
      without it, an unchanged bundle would be equally explained by a warm
      resident index simply never re-reading the column, not by
      `fused_score` ignoring it.
    """
    store, _adversarial_ids = _seed_adversarial_store(tmp_path)
    dim = store.embed_dim
    embedder = _DeterministicEmbedder(dim=dim)
    graph, assignment, rich_club = retrieve.build_runtime_graph(store)

    target_id = _find_seed_boundary_node(graph, store, embedder, _CENTRALITY_CUE)
    original_centrality = float(graph.get_centrality(target_id))

    baseline = _capture_recall(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_CENTRALITY_CUE, budget_tokens=1500,
    )
    assert str(target_id) not in [str(s) for s in baseline["seed_ids"]], (
        "the boundary node must not be a seed in the unmutated baseline"
    )

    # Positive arm: mutate the GRAPH PAYLOAD centrality only, well above the
    # corpus's natural near-zero scale.
    graph.set_node_centrality(target_id, 1.0)
    positive = _capture_recall(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_CENTRALITY_CUE, budget_tokens=1500,
    )
    assert str(target_id) in [str(s) for s in positive["seed_ids"]], (
        "boosting the boundary node's GRAPH PAYLOAD centrality must pull "
        "it into the seed set -- otherwise the seed-centrality source trap "
        "is vacuous"
    )
    assert _any_artifact_differs(baseline, positive), (
        "the graph-payload centrality mutation must change at least one "
        "captured artifact"
    )
    _assert_golden(
        storage_driver, "trap_centrality_source", "centrality_cue", positive, assignment,
    )

    # Revert the graph payload to its original value before the negative arm.
    graph.set_node_centrality(target_id, original_centrality)
    assert float(graph.get_centrality(target_id)) == pytest.approx(original_centrality)

    # Negative arm: write ONLY the store's `centrality` COLUMN, bypassing
    # the graph entirely (bare SQL, no hook fire either way -- this column
    # is never touched by `_fire_graph_sync_hook`), THEN force a cold
    # rank-index rebuild so the write is actually pulled into the Rust
    # engine's `Inner.centrality` field. Without the forced rebuild this
    # arm cannot discriminate "fused_score ignores centrality" from "the
    # warm resident index never re-reads the column" -- both would leave
    # the bundle unchanged for entirely different reasons.
    with store.db._conn_lock:
        store.db._conn.execute(
            "UPDATE records SET centrality = ? WHERE id = ?",
            (999.0, str(target_id)),
        )
    assert float(graph.get_centrality(target_id)) == pytest.approx(original_centrality), (
        "a store-column-only write must never reach the graph payload"
    )
    _reset_rank_index(store)
    negative = _capture_recall(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_CENTRALITY_CUE, budget_tokens=1500,
    )
    for key in _COMPARABLE_BUNDLE_KEYS:
        assert baseline[key] == negative[key], (
            f"a store-column-only centrality write, even after a cold "
            f"rank-index rebuild that loads it into Inner.centrality, "
            f"changed captured field {key!r} -- fused_score must not read "
            f"the store centrality column"
        )


def _insert_without_graph_or_exact_feed(store: MemoryStore, record: Any) -> None:
    """Insert `record` into the store's DB with the graph-sync hook AND the
    resident exact-index feed both suppressed for this one flush, so it
    lands in the SQL table without ever touching the graph payload or the
    resident exact-cosine matrix -- the store/exact-index divergence the
    publish-race trap depends on. Mirrors `_toggle_hook_insert`'s
    save/restore-in-finally idiom, extended to cover both feeds."""
    saved_hook = store._graph_sync_hook
    saved_feed = store._feed_exact_index
    store.register_graph_sync_hook(None)
    store._feed_exact_index = lambda *_a, **_kw: None
    try:
        store.insert(record)
        flush_record_buffer(store)
    finally:
        store.register_graph_sync_hook(saved_hook)
        store._feed_exact_index = saved_feed


def test_trap_authority_merge_publish_boundary(tmp_path, monkeypatch, storage_driver):
    """Trap E: the authority-merge boundary inside one `_capture_recall`
    call -- between `recall_for_response` and the harness's own
    `_build_authority_hits` -> `store.exact_top_k` call -- reads a
    consistent snapshot of the resident exact-cosine matrix even when a
    publish lands exactly at that boundary. This guards against a future
    snapshot/generation lever that would let the authority merge read a
    mid-publish exact index and tear the bundle. `_build_authority_hits`
    omits production's live-id-set liveness re-check, so this proves the
    harness's own merge boundary is read-consistent, not that production's
    liveness re-check would fail the same case.
    """
    store, _adversarial_ids = _seed_adversarial_store(tmp_path)
    dim = store.embed_dim
    embedder = _DeterministicEmbedder(dim=dim)
    graph, assignment, rich_club = retrieve.build_runtime_graph(store)

    # Warm the resident exact matrix from the base corpus, which does not
    # yet contain the injected record.
    _capture_recall(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_AUTHORITY_PUBLISH_CUE, budget_tokens=1500,
    )
    assert store._exact_index.is_warm is True, (
        "the resident exact matrix must warm from the baseline capture"
    )

    injected_rng = np.random.default_rng(_SEED + 95)
    injected = make_record(9105, injected_rng, dim, varied_tags=True, with_hv_payload=False)
    injected_vec = embedder.embed(_AUTHORITY_PUBLISH_CUE)
    injected.embedding = list(injected_vec)
    injected.literal_surface = (
        f"{_AUTHORITY_PUBLISH_CUE} carried by the injected authority record"
    )
    injected.tags = ["bench"]
    _stabilize_record(injected, 0, namespace=8)
    injected_id = injected.id

    _insert_without_graph_or_exact_feed(store, injected)
    assert str(injected_id) not in graph._node_payload, (
        "the injected record must never reach the graph payload"
    )
    assert store.get_batch([injected_id]).get(injected_id) is not None, (
        "the injected record must be hydratable from the store"
    )

    # Negative arm: a fresh no-wrapper capture must not surface the
    # injected id -- it is genuinely absent from the warm matrix.
    baseline_no_wrapper = _capture_recall(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_AUTHORITY_PUBLISH_CUE, budget_tokens=1500,
    )
    assert str(injected_id) not in [
        rid for rid, _s, _r in baseline_no_wrapper["merged_scores_reasons"]
    ], (
        "the injected record must be absent from the baseline authority "
        "merge -- otherwise the negative arm leaked it into the warm matrix"
    )
    assert store._exact_index.is_warm is True, (
        "the exact matrix must still be warm before installing the wrapper"
    )

    orig_exact_top_k = MemoryStore.exact_top_k
    fired = {"count": 0}
    fed_vec_holder: dict[str, list[float]] = {"vec": injected_vec}

    def wrapper(self, vec, k=10, *, build_if_cold=True):
        fired["count"] += 1
        assert self._exact_index.is_warm is True, (
            "the exact matrix must still be warm when the wrapper fires -- "
            "a cold index here would let build_if_cold rebuild vacuously"
        )
        self._feed_exact_index(str(injected_id), fed_vec_holder["vec"])
        return orig_exact_top_k(self, vec, k=k, build_if_cold=False)

    monkeypatch.setattr(MemoryStore, "exact_top_k", wrapper)

    # Positive arm: the mid-merge publish reaches the merge.
    wrapper_bundle_1 = _capture_recall(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_AUTHORITY_PUBLISH_CUE, budget_tokens=1500,
    )
    assert str(injected_id) in [
        rid for rid, _s, _r in wrapper_bundle_1["merged_scores_reasons"]
    ], (
        "the mid-merge publish must reach the authority merge -- otherwise "
        "the boundary wrapper is not firing where expected"
    )

    # Read-consistency: two captures with the wrapper installed are
    # byte-identical.
    wrapper_bundle_2 = _capture_recall(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_AUTHORITY_PUBLISH_CUE, budget_tokens=1500,
    )
    for key in _COMPARABLE_BUNDLE_KEYS:
        assert wrapper_bundle_1[key] == wrapper_bundle_2[key], (
            f"two captures with the mid-merge publish installed diverged on "
            f"{key!r} -- the authority-merge boundary is not read-consistent"
        )
    assert fired["count"] == 2, (
        "the wrapper must fire exactly once per capture -- a count other "
        "than 2 after two captures means the publish landed somewhere "
        "other than the intended boundary"
    )

    _assert_golden(
        storage_driver, "trap_authority_merge_publish", "authority_publish_cue",
        wrapper_bundle_1, assignment,
    )

    # Sensitivity: feed a NEGATED vector at the boundary -- the injected
    # record drops out of exact top-k -- and assert a captured artifact
    # changes, proving the injection channel is load-bearing.
    fed_vec_holder["vec"] = (-np.asarray(injected_vec, dtype=np.float32)).tolist()
    negated_bundle = _capture_recall(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_AUTHORITY_PUBLISH_CUE, budget_tokens=1500,
    )
    assert _any_artifact_differs(wrapper_bundle_1, negated_bundle), (
        "feeding a negated vector at the boundary produced IDENTICAL "
        "captures -- the injection channel would not be load-bearing"
    )


# ---------------------------------------------------------------------------
# Corpus replay: the differential + golden holds as corpus size grows.
# 3000 runs in the default single-file invocation; 8000/15000 are gated
# behind --runslow and skip the golden (differential byte-identity only).
# ---------------------------------------------------------------------------

_REPLAY_EDGE_FACTOR = 1.3
_REPLAY_CUE = "alice daily routine note 250"


def _seed_replay_corpus(store: MemoryStore, n: int, seed: int) -> None:
    """Build a stabilized (deterministic id/created_at) synthetic corpus of
    `n` records plus a bounded-cluster synthetic edge topology, the same
    shape `_seed_store`/`_seed_adversarial_store` use at 400 records, scaled
    up for the replay sweep."""
    dim = store.embed_dim
    rng = np.random.default_rng(seed)
    for i in range(n):
        rec = make_record(i, rng, dim, varied_tags=True, with_hv_payload=False)
        _stabilize_record(rec, i, namespace=7)
        store.insert(rec)
    flush_record_buffer(store)
    _build_synthetic_edges(store, n, _REPLAY_EDGE_FACTOR, seed)
    flush_edge_buffer(store)


@pytest.mark.parametrize(
    "n_records",
    [
        3000,
        pytest.param(8000, marks=pytest.mark.slow),
        pytest.param(15000, marks=pytest.mark.slow),
    ],
)
def test_corpus_replay_byte_parity(n_records, tmp_path, storage_driver):
    """The byte-differential + golden holds as corpus size grows. Only the
    3000-record size is golden-hashed here (unmarked, runs by default);
    8000/15000 are `--runslow`-gated and check differential byte-identity
    only.
    """
    store = MemoryStore(path=tmp_path / "replay-store")
    _seed_replay_corpus(store, n_records, seed=_SEED + n_records)

    graph, assignment, rich_club = retrieve.build_runtime_graph(store)
    embedder = _DeterministicEmbedder(dim=store.embed_dim)

    bundle = _capture_recall(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=_REPLAY_CUE, budget_tokens=1500,
    )
    assert bundle["rust_scorer_calls"] >= 1
    assert bundle["hits"], "the replay corpus cue must surface at least one hit"

    if n_records == 3000:
        _assert_golden(
            storage_driver, "corpus_replay", f"n{n_records}", bundle, assignment,
        )
    else:
        feed_bytes = _serialize_graph_feed(graph)
        reloaded_graph = _reload_graph_from_feed(feed_bytes)
        reloaded_bundle = _capture_recall(
            store=store, graph=reloaded_graph, assignment=assignment,
            rich_club=rich_club, embedder=embedder, cue=_REPLAY_CUE,
            budget_tokens=1500,
        )
        for key in _COMPARABLE_BUNDLE_KEYS:
            assert bundle[key] == reloaded_bundle[key], (
                f"replay at n={n_records}: snapshot/alternate-fed reader "
                f"diverged from the live reader on bundle field {key!r}"
            )
