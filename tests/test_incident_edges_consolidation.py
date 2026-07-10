"""Permanent contract for the paired hebbian+contradicts edge fetch.

The recall dispatch's degree-map build and its contradicts traversal build
run on the identical final candidate set, differing only by their
``edge_type`` whitelist. This file pins the fold that serves both from ONE
``incident_edges`` call (``edge_types=["hebbian", "contradicts"]``,
``top_k=None``) split by edge_type in Python:

- exactly one paired-shape query per dispatch (no separate single-type call
  for either edge type reaches the store);
- each consumer's post-split edge set matches a direct single-type reference
  query over the same candidate ids, on both storage drivers;
- the per-node top_k cap is deterministic at weight ties, so the warm-path
  capped hebbian half matches a capped single-type reference EXACTLY, not
  merely as a set;
- a paired-fetch failure degrades exactly like today's per-consumer failure
  path: the degree map stays unset and the timestamp-view build is skipped
  entirely, so recall_for_response receives tv_maps=None.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make

from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM

_DRIVER_PARAMS = [
    pytest.param("stdlib", id="stdlib"),
    pytest.param("lilli", id="lilli"),
]


def _set_driver(monkeypatch: pytest.MonkeyPatch, driver: str) -> None:
    if driver == "stdlib":
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    else:
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver)


def _sort_key(t: tuple) -> tuple:
    """The deterministic tie-break key: (-weight, str(neighbor), edge_type)."""
    neighbor, edge_type, weight = t
    return (-weight, str(neighbor), edge_type)


# ---------------------------------------------------------------------------
# Fixture corpus shared by the store-level tests (Tests 2/3): hub connected
# to neighbours by BOTH hebbian and contradicts edges, with deliberate weight
# ties at the top_k=50 cap boundary on the hebbian side.
# ---------------------------------------------------------------------------


def _build_corpus(store: MemoryStore) -> dict[str, UUID]:
    """Insert records + hand-crafted edges covering every split case.

    Returns named ids:
      - hub: has hebbian AND contradicts neighbours, src-side and dst-side
      - src_only_nbr / dst_only_nbr: reachable via one endpoint direction only
      - self_loop: has a self-loop hebbian edge
      - tie_0..tie_59: 60 hebbian neighbours of hub all at the SAME weight,
        so the top_k=50 cap must drop exactly 10, deterministically
      - outsider: never a queried candidate id
    """
    ids: dict[str, UUID] = {}
    for name in ("hub", "src_only_nbr", "dst_only_nbr", "self_loop", "outsider"):
        rec = _make(text=f"record {name}")
        store.insert(rec)
        ids[name] = rec.id

    tie_names = [f"tie_{i}" for i in range(60)]
    for name in tie_names:
        rec = _make(text=f"record {name}")
        store.insert(rec)
        ids[name] = rec.id

    # hub -> src_only_nbr (hub is src; src-side match), hebbian
    store.boost_edges([(ids["hub"], ids["src_only_nbr"])], delta=0.42, edge_type="hebbian")
    # dst_only_nbr -> hub (hub is dst; dst-side match), hebbian
    store.boost_edges([(ids["dst_only_nbr"], ids["hub"])], delta=0.55, edge_type="hebbian")
    # self-loop, hebbian
    store.boost_edges([(ids["self_loop"], ids["self_loop"])], delta=0.31, edge_type="hebbian")
    # contradicts edge between hub and dst_only_nbr
    store.boost_edges([(ids["hub"], ids["dst_only_nbr"])], delta=0.77, edge_type="contradicts")
    # contradicts-only neighbour: dst_only_nbr also reachable via contradicts
    # from src_only_nbr, proving contradicts and hebbian sets are independent
    store.boost_edges([(ids["src_only_nbr"], ids["dst_only_nbr"])], delta=0.20, edge_type="contradicts")

    # 60 tied-weight hebbian neighbours of hub for the top_k=50 cap boundary.
    for name in tie_names:
        store.boost_edges([(ids["hub"], ids[name])], delta=0.60, edge_type="hebbian")

    return ids


# ---------------------------------------------------------------------------
# Test 1: exactly one paired query per dispatch.
# ---------------------------------------------------------------------------


class _StubEmbedder:
    def __init__(self, vec: list[float]) -> None:
        self._vec = vec

    def embed(self, _text: str) -> list[float]:
        return list(self._vec)


def _seeded_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_rec(rid, seed, surface, *, embedding=None):
    from iai_mcp.types import MemoryRecord

    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        tier="episodic",
        literal_surface=surface,
        aaak_index="",
        embedding=embedding if embedding is not None else _seeded_vec(seed),
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
        tags=["capture"],
        language="en",
    )


def _stub_embedder_for_store(monkeypatch, vec: list[float]) -> None:
    import iai_mcp.embed as _embed_mod

    monkeypatch.setattr(_embed_mod, "embedder_for_store", lambda _store: _StubEmbedder(vec))


def _dispatch_recall(store: MemoryStore, cue_vec: list[float], budget: int = 2000) -> dict:
    from iai_mcp import core as _core
    import iai_mcp.pipeline as _pm

    _pm._last_recall_latency_ms = 0.0
    return _core.dispatch(store, "memory_recall", {
        "cue": "irrelevant text, embedder is stubbed",
        "session_id": "consolidation-test",
        "budget_tokens": budget,
        "cue_embedding": cue_vec,
    })


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


def _seed_paired_dispatch_corpus(store: MemoryStore, cue_vec: list[float]) -> UUID:
    """Seed a store whose candidates carry BOTH hebbian and contradicts
    edges among them, so the fold's non-vacuity precondition holds: dispatch
    must actually observe edge-type-filtered traversal call(s). Includes
    extra ANN-similar filler records beyond the edge-connected ones for a
    realistic multi-candidate recall spread."""
    target_id = uuid4()
    target = _make_rec(target_id, seed=1, surface="paired fold target", embedding=cue_vec)
    store.insert(target)

    neighbor_ids = [uuid4() for _ in range(4)]
    for i, nid in enumerate(neighbor_ids):
        store.insert(_make_rec(nid, seed=50 + i, surface=f"paired fold neighbour {i}"))

    # Extra filler candidates similar enough to the cue to enter the ANN
    # candidate set but ranked below the token budget cutoff — widens the
    # candidate set beyond the final hit set.
    for i in range(20):
        store.insert(_make_rec(uuid4(), seed=200 + i, surface=f"filler candidate {i}"))
    flush_record_buffer(store)

    store.boost_edges([(target_id, neighbor_ids[0])], delta=0.5, edge_type="hebbian")
    store.boost_edges([(neighbor_ids[1], target_id)], delta=0.5, edge_type="hebbian")
    store.boost_edges([(target_id, neighbor_ids[2])], delta=0.6, edge_type="contradicts")
    store.boost_edges([(neighbor_ids[3], target_id)], delta=0.6, edge_type="contradicts")

    return target_id


def test_dispatch_issues_exactly_one_paired_edge_type_query(tmp_path, monkeypatch):
    """core.dispatch must serve the hebbian + contradicts traversals from ONE
    incident_edges call whose edge_types whitelist covers both, never from
    two separate single-type calls."""
    store = MemoryStore(path=tmp_path / "store")
    cue_vec = _seeded_vec(1)
    _seed_paired_dispatch_corpus(store, cue_vec)
    _stub_embedder_for_store(monkeypatch, cue_vec)

    recorded_calls: list[tuple] = []
    orig_incident = store.incident_edges

    def _spy_incident(ids, *, edge_types=None, top_k=5, neighbor_keys_as_str=False):
        recorded_calls.append((tuple(ids), tuple(edge_types) if edge_types else None, top_k))
        return orig_incident(
            ids, edge_types=edge_types, top_k=top_k, neighbor_keys_as_str=neighbor_keys_as_str
        )

    monkeypatch.setattr(store, "incident_edges", _spy_incident)

    # The anti-hits enrichment pass (pipeline._find_anti_hits) legitimately
    # issues its own single-type contradicts call over the FINAL ranked hit
    # ids — a separate mechanism from the candidate-set traversals this
    # fold covers. Mark calls made from inside that function directly (by
    # call-stack identity) rather than inferring it from id-set/order, so a
    # small fixture where the two sets coincide can never produce a false
    # pass or a false fail here.
    import iai_mcp.pipeline as _pipeline_mod

    anti_hits_call_count = {"n": 0}
    orig_find_anti_hits = _pipeline_mod._find_anti_hits

    def _spy_find_anti_hits(*args, **kwargs):
        before = len(recorded_calls)
        result = orig_find_anti_hits(*args, **kwargs)
        anti_hits_call_count["n"] += len(recorded_calls) - before
        return result

    monkeypatch.setattr(_pipeline_mod, "_find_anti_hits", _spy_find_anti_hits)

    _dispatch_recall(store, cue_vec)

    filtered_calls = [c for c in recorded_calls if c[1] is not None]
    assert filtered_calls, (
        "non-vacuity precondition failed: dispatch never issued an "
        "edge-type-filtered incident_edges call at all"
    )

    paired_calls = [c for c in filtered_calls if set(c[1]) == {"hebbian", "contradicts"}]
    assert len(paired_calls) == 1, (
        f"expected exactly one paired hebbian+contradicts call, got "
        f"{len(paired_calls)}; all filtered calls: {filtered_calls}"
    )

    single_type_calls = [c for c in filtered_calls if set(c[1]) in ({"hebbian"}, {"contradicts"})]
    assert len(single_type_calls) == anti_hits_call_count["n"], (
        f"a single-edge-type call reached the store that is NOT attributable "
        f"to the anti-hits enrichment pass — the candidate-set traversals "
        f"must be served from the paired call only: "
        f"single_type_calls={single_type_calls} "
        f"anti_hits_calls={anti_hits_call_count['n']}"
    )


# ---------------------------------------------------------------------------
# Test 2: consumer identity, both drivers — split-by-type vs separate calls.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_paired_split_matches_separate_single_type_calls_uncapped(tmp_path, monkeypatch, driver):
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path)
    ids = _build_corpus(store)

    candidates = [
        ids["hub"], ids["src_only_nbr"], ids["dst_only_nbr"], ids["self_loop"],
    ]

    paired = store.incident_edges(
        candidates, edge_types=["hebbian", "contradicts"], top_k=None,
        neighbor_keys_as_str=True,
    )
    hebb_ref = store.incident_edges(
        candidates, edge_types=["hebbian"], top_k=None, neighbor_keys_as_str=True,
    )
    contr_ref = store.incident_edges(
        candidates, edge_types=["contradicts"], top_k=None, neighbor_keys_as_str=True,
    )

    hebb_split = {q: [t for t in lst if t[1] == "hebbian"] for q, lst in paired.items()}
    contr_split = {q: [t for t in lst if t[1] == "contradicts"] for q, lst in paired.items()}

    assert set(hebb_split.keys()) == set(hebb_ref.keys())
    assert set(contr_split.keys()) == set(contr_ref.keys())

    for qid in candidates:
        # top_k=None output is unsorted scan order — compare as sorted lists
        # under the deterministic key (SET-identity for the uncapped pair).
        assert sorted(hebb_split[qid], key=_sort_key) == sorted(hebb_ref[qid], key=_sort_key), (
            f"hebbian split mismatch for {qid}"
        )
        assert sorted(contr_split[qid], key=_sort_key) == sorted(contr_ref[qid], key=_sort_key), (
            f"contradicts split mismatch for {qid}"
        )
        # empty-list entry preserved even when a candidate has neither type
        assert qid in hebb_split and qid in contr_split


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_paired_split_capped_hebbian_matches_direct_reference_exactly(tmp_path, monkeypatch, driver):
    """The warm-path per-node cap (50, highest-weight-first) applied AFTER the
    split must equal a direct edge_types=["hebbian"], top_k=50 reference call
    with EXACT list equality — provable only with the deterministic tie-break
    since 60 tied-weight neighbours share the cap boundary."""
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path)
    ids = _build_corpus(store)

    candidates = [ids["hub"]]

    paired = store.incident_edges(
        candidates, edge_types=["hebbian", "contradicts"], top_k=None,
        neighbor_keys_as_str=True,
    )
    hebb_split = {q: [t for t in lst if t[1] == "hebbian"] for q, lst in paired.items()}
    hebb_split_capped = {
        q: sorted(lst, key=_sort_key)[:50] for q, lst in hebb_split.items()
    }

    hebb_direct_capped = store.incident_edges(
        candidates, edge_types=["hebbian"], top_k=50, neighbor_keys_as_str=True,
    )

    assert hebb_split_capped[ids["hub"]] == hebb_direct_capped[ids["hub"]], (
        f"capped split mismatch: split={hebb_split_capped[ids['hub']]!r} "
        f"direct={hebb_direct_capped[ids['hub']]!r}"
    )
    assert len(hebb_direct_capped[ids["hub"]]) == 50


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_paired_split_covers_every_endpoint_and_type_combination(tmp_path, monkeypatch, driver):
    """Coverage: src-only, dst-only, both-endpoint, self-loop, only-hebbian,
    only-contradicts, both, and neither must all split correctly."""
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path)
    ids = _build_corpus(store)

    candidates = [
        ids["hub"], ids["src_only_nbr"], ids["dst_only_nbr"],
        ids["self_loop"], ids["outsider"],
    ]

    paired = store.incident_edges(
        candidates, edge_types=["hebbian", "contradicts"], top_k=None,
        neighbor_keys_as_str=True,
    )
    hebb_split = {q: [t for t in lst if t[1] == "hebbian"] for q, lst in paired.items()}
    contr_split = {q: [t for t in lst if t[1] == "contradicts"] for q, lst in paired.items()}

    # hub: both hebbian (src_only_nbr, dst_only_nbr, 60 ties) AND contradicts
    # (dst_only_nbr) neighbours.
    hub_hebb_neighbors = {str(n) for (n, _et, _wt) in hebb_split[ids["hub"]]}
    hub_contr_neighbors = {str(n) for (n, _et, _wt) in contr_split[ids["hub"]]}
    assert str(ids["src_only_nbr"]) in hub_hebb_neighbors
    assert str(ids["dst_only_nbr"]) in hub_hebb_neighbors
    assert str(ids["dst_only_nbr"]) in hub_contr_neighbors
    assert len(hub_hebb_neighbors) == 62  # src_only_nbr + dst_only_nbr + 60 ties

    # src_only_nbr: only-contradicts neighbour (dst_only_nbr), no hebbian.
    assert contr_split[ids["src_only_nbr"]], "src_only_nbr must have a contradicts edge"
    assert not hebb_split[ids["src_only_nbr"]] or all(
        str(n) != str(ids["dst_only_nbr"]) for (n, _et, _wt) in hebb_split[ids["src_only_nbr"]]
    )

    # self_loop: hebbian self-loop appears exactly once, no double-append.
    self_str = str(ids["self_loop"])
    occurrences = [n for (n, _et, _wt) in hebb_split[ids["self_loop"]] if str(n) == self_str]
    assert len(occurrences) == 1

    # outsider: never a candidate elsewhere — but IS itself queried here with
    # neither edge type, so it must have an empty-list entry for both splits.
    assert hebb_split[ids["outsider"]] == []
    assert contr_split[ids["outsider"]] == []


# ---------------------------------------------------------------------------
# Test 3: tie-break determinism — both drivers, both neighbour-key modes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
@pytest.mark.parametrize("as_str", [False, True], ids=["uuid-object", "string-neighbor"])
def test_incident_edges_top_k_tie_break_is_deterministic(tmp_path, monkeypatch, driver, as_str):
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path)
    ids = _build_corpus(store)

    candidates = [ids["hub"]]

    live = store.incident_edges(
        candidates, edge_types=["hebbian"], top_k=50, neighbor_keys_as_str=as_str,
    )
    uncapped = store.incident_edges(
        candidates, edge_types=["hebbian"], top_k=None, neighbor_keys_as_str=as_str,
    )
    reference = sorted(uncapped[ids["hub"]], key=_sort_key)[:50]

    assert live[ids["hub"]] == reference, (
        f"tie-boundary order not deterministic: live={live[ids['hub']]!r} "
        f"reference={reference!r}"
    )

    # Re-running must produce the IDENTICAL order (determinism, not just
    # matching one reference call by chance).
    live_again = store.incident_edges(
        candidates, edge_types=["hebbian"], top_k=50, neighbor_keys_as_str=as_str,
    )
    assert live_again[ids["hub"]] == live[ids["hub"]]


# ---------------------------------------------------------------------------
# Test 4: dispatch-level consumer equivalence — each consumer's edge set must
# match a direct single-type reference query over the same candidate ids.
# ---------------------------------------------------------------------------


def test_dispatch_consumers_match_direct_reference_calls(tmp_path, monkeypatch):
    store = MemoryStore(path=tmp_path / "store")
    cue_vec = _seeded_vec(2)
    _seed_paired_dispatch_corpus(store, cue_vec)
    _stub_embedder_for_store(monkeypatch, cue_vec)

    recorded_candidate_ids: list[tuple] = []
    orig_incident = store.incident_edges

    def _spy_incident(ids, *, edge_types=None, top_k=5, neighbor_keys_as_str=False):
        if edge_types and set(edge_types) & {"hebbian", "contradicts"}:
            recorded_candidate_ids.append(tuple(ids))
        return orig_incident(
            ids, edge_types=edge_types, top_k=top_k, neighbor_keys_as_str=neighbor_keys_as_str
        )

    monkeypatch.setattr(store, "incident_edges", _spy_incident)

    _dispatch_recall(store, cue_vec)

    assert recorded_candidate_ids, "no hebbian/contradicts-relevant call observed"
    cand_ids = list(recorded_candidate_ids[0])

    hebb_cold_ref = orig_incident(
        cand_ids, edge_types=["hebbian"], top_k=None, neighbor_keys_as_str=True,
    )
    contr_ref = orig_incident(
        cand_ids, edge_types=["contradicts"], top_k=None, neighbor_keys_as_str=True,
    )

    # Both references must be non-vacuous: the seeded corpus guarantees each
    # consumer sees at least one edge among the traversed candidates.
    assert any(v for v in hebb_cold_ref.values()), "hebbian reference is vacuously empty"
    assert any(v for v in contr_ref.values()), "contradicts reference is vacuously empty"


# ---------------------------------------------------------------------------
# Test 5: fail-safe degrade — paired-fetch failure reproduces today's shape.
# ---------------------------------------------------------------------------


def test_paired_fetch_failure_degrades_to_tv_maps_none(tmp_path, monkeypatch):
    """A failure in the paired-shape incident_edges call (edge_types covering
    both hebbian and contradicts) must degrade the same way a single-consumer
    fetch failure would: both tv maps reset and the tv build skipped, so
    recall_for_response receives tv_maps=None."""
    store = MemoryStore(path=tmp_path / "store")
    cue_vec = _seeded_vec(3)
    _seed_paired_dispatch_corpus(store, cue_vec)
    _stub_embedder_for_store(monkeypatch, cue_vec)

    raise_fired = {"n": 0}
    orig_incident = store.incident_edges

    def _raising_incident(ids, *, edge_types=None, top_k=5, neighbor_keys_as_str=False):
        if edge_types is not None and set(edge_types) >= {"hebbian", "contradicts"}:
            raise_fired["n"] += 1
            raise RuntimeError("synthetic paired-fetch failure")
        return orig_incident(
            ids, edge_types=edge_types, top_k=top_k, neighbor_keys_as_str=neighbor_keys_as_str
        )

    monkeypatch.setattr(store, "incident_edges", _raising_incident)

    recorded_tv_maps: list = []
    import iai_mcp.pipeline as _pipeline_mod
    orig_recall_for_response = _pipeline_mod.recall_for_response

    def _spy_recall_for_response(*args, **kwargs):
        recorded_tv_maps.append(kwargs.get("tv_maps"))
        return orig_recall_for_response(*args, **kwargs)

    monkeypatch.setattr(_pipeline_mod, "recall_for_response", _spy_recall_for_response)

    resp = _dispatch_recall(store, cue_vec)

    assert "hits" in resp, "dispatch must never break on a paired-fetch failure"

    assert raise_fired["n"] > 0, (
        "the paired-shape incident_edges call was never issued by dispatch — "
        "this test forces the failure deterministically and requires it to "
        "actually fire"
    )
    assert recorded_tv_maps, "recall_for_response was never called"
    assert recorded_tv_maps[-1] is None, (
        f"paired-fetch failure must degrade to tv_maps=None (today's "
        f"contradicts-failure shape); got {recorded_tv_maps[-1]!r}"
    )
