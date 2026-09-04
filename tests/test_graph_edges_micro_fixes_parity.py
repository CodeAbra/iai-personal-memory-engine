"""Byte-identity guards for the ge_incident row-handling micro-fixes on the
graph-edges dispatch path (``core/__init__.py``):

1. the hebbian/contradicts split over ``store.incident_edges``'s fetched
   result collapses from two dict-comprehension re-scans into one inline
   bucketed pass;
2. the contradicts-destination ``get_batch`` uses ``decode="rank"``;
3. ``_incident_edges_warm`` constructs ``UUID()`` only for the top_k
   survivors instead of over the full adjacency.

Part A pins the pre-fix ``_incident_edges_warm`` body as a reference
implementation and diffs it against the shipped function directly (pure,
in-memory, no store/driver needed). Part B exercises the two inline
dispatch-level fixes end to end through ``core.dispatch`` on both storage
drivers, using structural spies (call-count / kwarg capture) to prove the
fixes landed and first-principles known-topology assertions (a hub of a
known true degree) to prove the served set and the ranking degree term are
unaffected -- a direct old-vs-new run is not possible for code inlined in a
single dispatch function, so structural + ground-truth assertions are the
falsifiable substitute (see the plan's Interfaces section).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import MemoryRecord
from tests._helpers import stub_embedder_for_store

_DIM = 16
_FIXED_CREATED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)
_HEBB_TRAVERSAL_CAP = 50  # mirrors core/__init__.py's _HEBB_TRAVERSAL_CAP


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _small_embed_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_DIM))


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


def _select_driver(driver: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")


def _assert_driver_matches(store: MemoryStore, driver: str) -> None:
    """Hard proof the ``driver`` parametrization actually selected the
    requested engine — ``col_index_ready`` is a lilli-engine-only PyO3
    binding absent from the stdlib connection wrapper entirely (no
    ColIndex/reader-pool mechanism exists on that driver at all)."""
    has_lilli_binding = hasattr(store.db._conn, "col_index_ready")
    if driver == "lilli":
        assert has_lilli_binding, (
            "driver='lilli' requested but store.db._conn has no "
            "col_index_ready binding — the lilli engine was not selected"
        )
    else:
        assert not has_lilli_binding, (
            "driver='stdlib' requested but store.db._conn exposes "
            "col_index_ready — the lilli engine was selected instead"
        )


@pytest.fixture
def _store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _make(driver: str, suffix: str = "") -> MemoryStore:
        _select_driver(driver, monkeypatch)
        store = MemoryStore(path=tmp_path / f"store-{driver}{suffix}")
        _assert_driver_matches(store, driver)
        return store
    return _make


class _StubEmbedder:
    """Deterministic recall embedder double — a fixed vector regardless of text."""

    def __init__(self, vec: list[float]) -> None:
        self._vec = vec

    def embed(self, _text: str) -> list[float]:
        return list(self._vec)


def _seeded_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _unit_vector_with_cosine(cue_vec: list[float], target_cos: float, seed: int) -> list[float]:
    """Unit vector at an exact known cosine to ``cue_vec`` — used to place a
    record deterministically inside or outside the ANN K_CANDIDATES cutoff."""
    cue = np.asarray(cue_vec, dtype=np.float32)
    cue = cue / float(np.linalg.norm(cue))
    rng = np.random.default_rng(seed)
    probe = rng.standard_normal(_DIM).astype(np.float32)
    probe = probe - float(np.dot(cue, probe)) * cue
    probe_norm = float(np.linalg.norm(probe))
    if probe_norm > 0:
        probe = probe / probe_norm
    alpha = float(target_cos)
    beta = float(max(0.0, 1.0 - alpha * alpha)) ** 0.5
    v = alpha * cue + beta * probe
    n = float(np.linalg.norm(v))
    if n > 0:
        v = v / n
    return v.astype(np.float32).tolist()


def _stub_embedder_for_store(monkeypatch: pytest.MonkeyPatch, vec: list[float]) -> None:
    stub_embedder_for_store(monkeypatch, _StubEmbedder(vec))


def _make_rec(
    rid: UUID, seed: int, surface: str, *,
    embedding: list[float] | None = None,
    created_at: datetime | None = None,
) -> MemoryRecord:
    ts = created_at or _FIXED_CREATED_AT
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
        created_at=ts,
        updated_at=ts,
        tags=["capture"],
        language="en",
    )


# ---------------------------------------------------------------------------
# Part A: _incident_edges_warm — pinned reference vs shipped implementation
# ---------------------------------------------------------------------------


def _incident_edges_warm_reference(store, ids: list, top_k: "int | None" = 5) -> dict:
    """Pinned copy of the pre-fix ``_incident_edges_warm`` body
    (``core/__init__.py``, before the UUID-construction fix). Kept verbatim
    as the byte-identity oracle for Part A."""
    memo = getattr(store, "_warm_graph_bundle", None)
    graph = memo[0][0] if memo is not None else None
    adj = getattr(graph, "_adj", None) if graph is not None else None
    if not adj:
        return store.incident_edges(ids, top_k=top_k)
    result: dict = {}
    for rid in ids:
        nbrs = adj.get(str(rid))
        if not nbrs:
            result[rid] = []
            continue
        items = list(nbrs.items())
        rows = []
        for nbr_label, attrs in items:
            try:
                nbr = UUID(nbr_label)
            except (ValueError, AttributeError):
                continue
            rows.append((
                nbr,
                str(attrs.get("edge_type", "hebbian")),
                float(attrs.get("weight", 1.0)),
            ))
        rows.sort(key=lambda t: (-t[2], str(t[0]), t[1]))
        result[rid] = rows[:top_k] if top_k is not None else rows
    return result


def _canonical_label(n: int) -> str:
    return str(UUID(int=n))


def _fake_store_with_adj(adj: dict):
    graph_obj = SimpleNamespace(_adj=adj)
    bundle = (graph_obj, None, None)
    return SimpleNamespace(_warm_graph_bundle=(bundle, None, None, None, None))


def test_incident_edges_warm_hub_binding_cap_byte_identical():
    """Hub with degree 8 > top_k=5, distinct weights: the cap is binding and
    the survivors + order must match the reference exactly."""
    from iai_mcp.core import _incident_edges_warm

    hub_id = UUID(int=1)
    nbrs = {
        _canonical_label(i): {
            "edge_type": "hebbian" if i % 2 == 0 else "contradicts",
            "weight": float(i) * 0.1,
        }
        for i in range(2, 10)
    }
    store = _fake_store_with_adj({str(hub_id): nbrs})

    ref = _incident_edges_warm_reference(store, [hub_id], top_k=5)
    new = _incident_edges_warm(store, [hub_id], top_k=5)

    assert new == ref
    assert len(new[hub_id]) == 5, "cap must be binding on this fixture"


def test_incident_edges_warm_weight_ties_tiebreak_byte_identical():
    """All-tied weights force the tiebreak (label, then edge_type) to fire —
    the raw-label sort must reproduce it exactly for canonical labels."""
    from iai_mcp.core import _incident_edges_warm

    hub_id = UUID(int=1)
    nbrs = {
        _canonical_label(i): {"edge_type": "hebbian", "weight": 1.0}
        for i in range(2, 8)
    }
    store = _fake_store_with_adj({str(hub_id): nbrs})

    ref = _incident_edges_warm_reference(store, [hub_id], top_k=None)
    new = _incident_edges_warm(store, [hub_id], top_k=None)

    assert new == ref


def test_incident_edges_warm_non_canonical_length_label_dropped_control():
    """Non-vacuity control (membership): a 32-hex no-hyphen label is
    UUID()-parseable (old code accepts it) but fails the 36-char canonical
    shape (new code's filter drops it). Proves the filter is not a no-op.
    Canonical siblings in the SAME adjacency must still agree exactly."""
    from iai_mcp.core import _incident_edges_warm

    hub_id = UUID(int=1)
    canonical_lbl = _canonical_label(2)
    malformed_lbl = "a" * 32  # UUID()-parseable, 32 chars — not the 36-char canonical shape
    nbrs = {
        canonical_lbl: {"edge_type": "hebbian", "weight": 1.0},
        malformed_lbl: {"edge_type": "hebbian", "weight": 2.0},
    }
    store = _fake_store_with_adj({str(hub_id): nbrs})

    ref = _incident_edges_warm_reference(store, [hub_id], top_k=None)
    new = _incident_edges_warm(store, [hub_id], top_k=None)

    ref_ids = {t[0] for t in ref[hub_id]}
    new_ids = {t[0] for t in new[hub_id]}
    assert UUID(malformed_lbl) in ref_ids, "reference (old) behavior must include the malformed label"
    assert UUID(malformed_lbl) not in new_ids, "new behavior must drop the malformed label"

    canonical_uuid = UUID(canonical_lbl)
    ref_canon = [t for t in ref[hub_id] if t[0] == canonical_uuid]
    new_canon = [t for t in new[hub_id] if t[0] == canonical_uuid]
    assert ref_canon == new_canon, "the canonical sibling entry must be byte-identical"


def test_incident_edges_warm_uppercase_hex_label_documented_unreachable_divergence():
    """Documents a real gap in the plan's byte-identity claim: an uppercase-
    hex 36-char label PASSES the shipped filter (``_is_canonical_uuid_str``'s
    hex set includes A-F) but sorts differently raw vs canonicalized, so the
    reference and the new code CAN disagree on order for such a label.

    This is provably unreachable in production: every ``_adj`` key is
    ``str(UUID_obj)`` by construction (``graph.py`` ``add_node``/``add_edge``
    at the ``str(node_id)`` call sites, fed by ``UUID(row[...])`` in
    ``retrieve.py``'s node/edge streaming, itself sourced from columns
    written through ``_uuid_literal``'s lowercase-only regex) — a mixed-case
    label can never reach this function's adjacency in the live system. The
    test asserts the divergence exists (so a future change to the filter
    that silently "fixes" this is visible) rather than pretending it away.
    """
    from iai_mcp.core import _incident_edges_warm

    hub_id = UUID(int=1)
    upper_lbl = "FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF"
    lower_lbl = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    nbrs = {
        upper_lbl: {"edge_type": "hebbian", "weight": 1.0},
        lower_lbl: {"edge_type": "hebbian", "weight": 1.0},
    }
    store = _fake_store_with_adj({str(hub_id): nbrs})

    ref = _incident_edges_warm_reference(store, [hub_id], top_k=None)
    new = _incident_edges_warm(store, [hub_id], top_k=None)

    # Every survivor is returned as a UUID object, so str(t[0]) is ALWAYS the
    # lowercase canonical form regardless of the source label's case — order
    # is what carries the divergence signal here, not the string content.
    lower_str = str(UUID(lower_lbl))
    upper_str = str(UUID(upper_lbl))
    ref_order = [str(t[0]) for t in ref[hub_id]]
    new_order = [str(t[0]) for t in new[hub_id]]
    # Reference sorts on str(UUID(label)) — both normalize to lowercase, and
    # "aaaa..." < "ffff..." alphabetically, so the lower-origin entry is first.
    assert ref_order == [lower_str, upper_str]
    # New code sorts on the RAW label string — ASCII 'F' (0x46) < 'a' (0x61),
    # so the raw-uppercase-origin entry sorts FIRST, the opposite order.
    assert new_order == [upper_str, lower_str]
    assert ref_order != new_order, (
        "this divergence is expected for a hypothetical uppercase label; "
        "it never occurs in production because _adj keys are always "
        "str(UUID_obj) (always lowercase canonical) by construction"
    )


def test_incident_edges_warm_top_k_none_returns_all_entries():
    from iai_mcp.core import _incident_edges_warm

    hub_id = UUID(int=1)
    nbrs = {
        _canonical_label(i): {"edge_type": "hebbian", "weight": float(i)}
        for i in range(2, 6)
    }
    store = _fake_store_with_adj({str(hub_id): nbrs})

    ref = _incident_edges_warm_reference(store, [hub_id], top_k=None)
    new = _incident_edges_warm(store, [hub_id], top_k=None)

    assert new == ref
    assert len(new[hub_id]) == 4


# ---------------------------------------------------------------------------
# Part B: dispatch-level structural + degree/served-set fixtures
# ---------------------------------------------------------------------------


def _build_layer1_fixture(
    store: MemoryStore, cue_vec: list[float], *,
    hub_degree: int = 60, with_contradicts: bool = True, noise_count: int = 0,
):
    """Hub record whose embedding equals ``cue_vec`` (guaranteed top ANN
    candidate) plus ``hub_degree`` hebbian neighbours at weight 1.0 (true
    degree exceeds ``_HEBB_TRAVERSAL_CAP`` on purpose). Optional low-weight
    (0.01) contradicts neighbours: weight 0.01 loses every top-5 hop1/hop2
    spread sort against the weight-1.0 hebbian edges, so the contradicts
    targets are NEVER absorbed into the candidate set by spread.

    ``noise_count`` inserts extra filler records at a cosine ABOVE the
    contradicts targets' -0.9 (used only when ``with_contradicts`` needs the
    targets kept out of ANN's own K_CANDIDATES=200 cutoff too — a tiny corpus
    under 200 records makes ANN return everyone regardless of cosine, which
    would defeat the "only reachable via the contradicts fetch" guarantee).
    """
    hub_id = uuid4()
    store.insert(_make_rec(hub_id, seed=0, surface="hub schema record", embedding=cue_vec))

    hebb_pairs = []
    for i in range(hub_degree):
        nid = uuid4()
        emb = _unit_vector_with_cosine(cue_vec, 0.05, seed=1000 + i) if with_contradicts else None
        store.insert(_make_rec(nid, seed=1000 + i, surface=f"hebbian filler {i}", embedding=emb))
        hebb_pairs.append((hub_id, nid))
    if hebb_pairs:
        store.boost_edges(hebb_pairs, edge_type="hebbian", delta=1.0)

    for i in range(noise_count):
        nid = uuid4()
        emb = _unit_vector_with_cosine(cue_vec, 0.05, seed=4000 + i)
        store.insert(_make_rec(nid, seed=4000 + i, surface=f"noise filler {i}", embedding=emb))

    contr_ids: list[UUID] = []
    contr_created_at: dict[UUID, datetime] = {}
    if with_contradicts:
        contr_pairs = []
        for i in range(2):
            cid = uuid4()
            ts = datetime(2020, 6, 15 + i, 12, 0, 0, tzinfo=timezone.utc)
            emb = _unit_vector_with_cosine(cue_vec, -0.9, seed=2000 + i)
            store.insert(_make_rec(
                cid, seed=2000 + i,
                surface=f"contradicts target {i} distinct surface text",
                embedding=emb, created_at=ts,
            ))
            contr_ids.append(cid)
            contr_created_at[cid] = ts
            contr_pairs.append((hub_id, cid))
        store.boost_edges(contr_pairs, edge_type="contradicts", delta=0.01)

    flush_record_buffer(store)
    return hub_id, contr_ids, contr_created_at


def _dispatch_recall(store: MemoryStore, cue_vec: list[float], session_id: str) -> dict:
    from iai_mcp import core as _core
    import iai_mcp.pipeline as _pipeline_mod

    store._build_exact_index_sync()
    _pipeline_mod._last_recall_latency_ms = 0.0
    return _core.dispatch(store, "memory_recall", {
        "cue": "layer1 micro fix probe",
        "session_id": session_id,
        "budget_tokens": 2000,
        "cue_embedding": cue_vec,
    })


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_paired_edge_fetch_split_is_single_pass(driver, _store, monkeypatch):
    """The hebbian/contradicts split must scan the fetched paired-edges dict
    exactly ONCE (single inline bucketed pass), not twice (two dict
    comprehensions each re-scanning the same result) -- AND both buckets it
    produces must carry the right content: the hub's hebbian bucket must
    still drive a degree of 60 (not 0, not double-counted), and its
    contradicts bucket must still carry exactly the 2 known contradicts
    destinations (not merged with the hebbian bucket, not dropped)."""
    store = _store(driver)
    cue_vec = _seeded_vec(1)
    hub_id, contr_ids, _ = _build_layer1_fixture(
        store, cue_vec, hub_degree=60, with_contradicts=True, noise_count=200,
    )
    _stub_embedder_for_store(monkeypatch, cue_vec)

    orig_incident_edges = store.incident_edges
    items_calls: list[int] = []

    def _spy_incident_edges(*args, **kwargs):
        result = orig_incident_edges(*args, **kwargs)
        if kwargs.get("edge_types") == ["hebbian", "contradicts"]:
            class _CountingDict(dict):
                def items(self_inner):
                    items_calls.append(1)
                    return dict.items(self_inner)
            return _CountingDict(result)
        return result

    monkeypatch.setattr(store, "incident_edges", _spy_incident_edges)

    import iai_mcp.pipeline as _pl
    captured: dict = {}
    orig_rfr = _pl.recall_for_response

    def _spy_rfr(*args, **kwargs):
        g = kwargs.get("graph")
        captured["global_degree"] = dict(getattr(g, "_global_degree", {}) or {})
        captured["tv_maps"] = kwargs.get("tv_maps")
        return orig_rfr(*args, **kwargs)

    monkeypatch.setattr(_pl, "recall_for_response", _spy_rfr)

    _dispatch_recall(store, cue_vec, "layer1-single-pass")

    assert items_calls, "the paired-edges fetch (edge_types=['hebbian','contradicts']) was never observed"
    assert len(items_calls) == 1, (
        f"expected exactly ONE .items() pass over the fetched paired edges "
        f"(single-pass bucketed split), got {len(items_calls)} passes"
    )

    # Hebbian bucket content, from the SAME single pass: the hub's degree
    # must still be the true unbounded count.
    assert captured.get("global_degree", {}).get(str(hub_id)) == 60, (
        f"single-pass hebbian bucket must still drive degree 60 for the hub, "
        f"got {captured.get('global_degree')}"
    )

    # Contradicts bucket content, from the SAME single pass: tv_outgoing for
    # the hub must carry exactly the 2 known contradicts destinations, not
    # merged with hebbian neighbors and not dropped.
    tv_maps = captured.get("tv_maps")
    assert tv_maps is not None, "tv_maps must reach recall_for_response"
    tv_outgoing, _tv_ts = tv_maps
    hub_outgoing = set(tv_outgoing.get(str(hub_id), []))
    assert hub_outgoing == {str(c) for c in contr_ids}, (
        f"single-pass contradicts bucket must carry exactly the known "
        f"contradicts destinations {contr_ids}; got {hub_outgoing}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_contradicts_get_batch_uses_rank_decode_with_field_integrity(driver, _store, monkeypatch):
    """The contradicts-destination get_batch must use decode='rank', and the
    rank-decoded records must carry the exact stored id + created_at (the
    only two fields this call site consumes) — proving decode='rank' does
    not silently corrupt the served set for this fetch."""
    monkeypatch.delenv("IAI_MCP_LAZY_DECODE_OFF", raising=False)
    store = _store(driver)
    cue_vec = _seeded_vec(2)
    # noise_count pushes the corpus past ANN's K_CANDIDATES=200 so the
    # contradicts targets (cosine -0.9) rank outside the ANN candidate set
    # and are reachable ONLY through the layer1 contradicts-destination
    # fetch this test targets — a tiny corpus would make ANN return
    # everyone regardless of cosine and short-circuit the test.
    _hub_id, contr_ids, contr_created_at = _build_layer1_fixture(
        store, cue_vec, hub_degree=60, with_contradicts=True, noise_count=200,
    )
    _stub_embedder_for_store(monkeypatch, cue_vec)

    orig_get_batch = store.get_batch
    calls: list[tuple[set, str, dict]] = []

    def _spy_get_batch(ids, **kwargs):
        result = orig_get_batch(ids, **kwargs)
        calls.append((set(ids), kwargs.get("decode", "full"), result))
        return result

    monkeypatch.setattr(store, "get_batch", _spy_get_batch)

    _dispatch_recall(store, cue_vec, "layer1-rank-decode")

    matching = [c for c in calls if set(contr_ids) <= c[0]]
    assert matching, (
        f"expected a get_batch call covering the contradicts-destination ids "
        f"{contr_ids}; observed calls={[(ids, dec) for ids, dec, _ in calls]}"
    )
    ids_set, decode_used, result = matching[0]
    assert decode_used == "rank", (
        f"contradicts-destination get_batch must use decode='rank', got {decode_used!r}"
    )

    for cid in contr_ids:
        assert cid in result, f"{cid} missing from rank-decode get_batch result"
        rec = result[cid]
        assert rec.id == cid
        assert rec.created_at == contr_created_at[cid], (
            f"decode='rank' must preserve created_at exactly: "
            f"expected {contr_created_at[cid]}, got {rec.created_at}"
        )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_cold_path_degree_is_unbounded_true_count(driver, _store, monkeypatch):
    """Cold path (no persisted per-node degree cache): the ranking degree
    term MUST be len() of the unbounded hebbian traversal — the hub's true
    degree (60), never the _HEBB_TRAVERSAL_CAP (50)."""
    store = _store(driver)
    cue_vec = _seeded_vec(3)
    hub_id, _, _ = _build_layer1_fixture(store, cue_vec, hub_degree=60, with_contradicts=False)
    _stub_embedder_for_store(monkeypatch, cue_vec)

    import iai_mcp.pipeline as _pl
    captured: dict = {}
    orig_rfr = _pl.recall_for_response

    def _spy_rfr(*args, **kwargs):
        g = kwargs.get("graph")
        captured["global_degree"] = dict(getattr(g, "_global_degree", {}) or {})
        captured["max_degree"] = getattr(g, "_max_degree", None)
        return orig_rfr(*args, **kwargs)

    monkeypatch.setattr(_pl, "recall_for_response", _spy_rfr)

    _dispatch_recall(store, cue_vec, "layer1-cold-degree")

    assert str(hub_id) in captured.get("global_degree", {}), captured
    assert captured["global_degree"][str(hub_id)] == 60, (
        f"cold-path degree must be the TRUE unbounded hebbian count (60), "
        f"not the traversal cap ({_HEBB_TRAVERSAL_CAP}); "
        f"got {captured['global_degree'][str(hub_id)]}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_warm_path_degree_uses_cached_true_count_despite_traversal_cap(driver, _store, monkeypatch):
    """Warm path (persisted per-node degree cache present): the ranking
    degree term MUST still be the hub's cached TRUE degree (60), not the
    bounded traversal count (capped at 50) — the cache is what lets the
    warm-path latency bound coexist with correct ranking."""
    store = _store(driver)
    cue_vec = _seeded_vec(4)
    hub_id, _, _ = _build_layer1_fixture(store, cue_vec, hub_degree=60, with_contradicts=False)

    from iai_mcp import runtime_graph_cache as _rgc
    from iai_mcp.retrieve import build_runtime_graph

    graph0, assignment, rc = build_runtime_graph(store)
    max_degree = max((d for _, d in graph0.degrees()), default=0)
    node_degrees = {nid: d for nid, d in graph0.degrees()}
    assert node_degrees.get(hub_id) == 60, (
        f"fixture sanity: pre-cache graph degree for the hub must be 60, "
        f"got {node_degrees.get(hub_id)}"
    )
    _rgc.save(store, assignment, rc, max_degree=max_degree, node_degrees=node_degrees)

    _stub_embedder_for_store(monkeypatch, cue_vec)

    import iai_mcp.pipeline as _pl
    captured: dict = {}
    orig_rfr = _pl.recall_for_response

    def _spy_rfr(*args, **kwargs):
        g = kwargs.get("graph")
        captured["global_degree"] = dict(getattr(g, "_global_degree", {}) or {})
        return orig_rfr(*args, **kwargs)

    monkeypatch.setattr(_pl, "recall_for_response", _spy_rfr)

    _dispatch_recall(store, cue_vec, "layer1-warm-degree")

    assert str(hub_id) in captured.get("global_degree", {}), captured
    assert captured["global_degree"][str(hub_id)] == 60, (
        "warm-path degree must be the cached TRUE degree (60), not the "
        f"traversal-cap-bounded value; got {captured['global_degree'][str(hub_id)]}"
    )
