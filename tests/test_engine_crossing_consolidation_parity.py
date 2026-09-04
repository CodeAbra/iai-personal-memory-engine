"""Infrastructure scaffold for the engine-crossing-consolidation kill switch.

``IAI_MCP_CROSSING_CONSOLIDATION_OFF=1`` forces the legacy pre-consolidation
recall path; unset selects the new consolidated path -- mirroring the
project's existing bench-kill-switch convention (``IAI_MCP_LAZY_DECODE_OFF``,
``IAI_MCP_RO_POOL_OFF``). Nothing is wired to the switch yet: this file only
proves the on/off harness toggles cleanly in one process on both drivers, so
later plans can add byte-identical per-finding assertions against a stable
scaffold. Default-gate (no slow/perf marker) so it participates in every
per-task -k verify and the phase's stdlib-skip-count baseline.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.core import dispatch
from iai_mcp.graph import MemoryGraph
from iai_mcp.pask_teachback import verify_hit_set
from iai_mcp.pipeline import _crossing_consolidation_off, _find_anti_hits, recall_for_response
from iai_mcp.store import MemoryStore, flush_edge_buffer, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryHit, MemoryRecord
from tests._helpers import stub_embedder_for_store
from tests.test_recall_core_unit import _build_store_and_graph, _flat_assignment, _make

CROSSING_KILL_SWITCH_ENV = "IAI_MCP_CROSSING_CONSOLIDATION_OFF"


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
            pytest.skip("iai_mcp_native not built -- lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


class _NullEmbedder:
    """Never called: the smoke recall supplies cue_embedding directly."""

    DIM = EMBED_DIM

    def embed(self, text: str) -> list[float]:
        raise AssertionError("embedder.embed() called -- cue_embedding should have bypassed it")


def test_crossing_consolidation_off_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CROSSING_KILL_SWITCH_ENV, raising=False)
    assert _crossing_consolidation_off() is False
    monkeypatch.setenv(CROSSING_KILL_SWITCH_ENV, "1")
    assert _crossing_consolidation_off() is True
    monkeypatch.setenv(CROSSING_KILL_SWITCH_ENV, "0")
    assert _crossing_consolidation_off() is False


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_crossing_kill_switch_toggles_cleanly_in_one_process(
    driver: str, tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The switch flips cleanly in one process against the same store: a
    recall completes and returns hits with the switch OFF (new path) then ON
    (legacy path). Infrastructure only -- per-finding byte-identical
    assertions land once plans 02/03 wire consolidations behind the switch.
    """
    _select_driver(driver, monkeypatch)
    store, graph, recs = _build_store_and_graph(tmp_path / f"store-{driver}", n=5)
    flush_record_buffer(store)
    store._build_exact_index_sync()
    assignment = _flat_assignment(recs)
    cue_vec = np.array(recs[0].embedding, dtype=np.float32)

    monkeypatch.delenv(CROSSING_KILL_SWITCH_ENV, raising=False)
    assert _crossing_consolidation_off() is False
    resp_new = recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=[],
        embedder=_NullEmbedder(), cue="crossing parity scaffold probe",
        session_id="crossing-parity-scaffold", budget_tokens=2000,
        cue_embedding=cue_vec.tolist(),
    )
    assert resp_new.hits, "switch OFF (new path) produced no hits"

    monkeypatch.setenv(CROSSING_KILL_SWITCH_ENV, "1")
    assert _crossing_consolidation_off() is True
    resp_legacy = recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=[],
        embedder=_NullEmbedder(), cue="crossing parity scaffold probe",
        session_id="crossing-parity-scaffold", budget_tokens=2000,
        cue_embedding=cue_vec.tolist(),
    )
    assert resp_legacy.hits, "switch ON (legacy path) produced no hits"


_ENRICH_FIELDS = ("session_id", "captured_at", "epistemic_status", "salience_level")


def _enrich_fields(hit) -> tuple:
    return tuple(getattr(hit, f) for f in _ENRICH_FIELDS)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_enrichment_field_parity_across_kill_switch_with_budget_drop(
    driver: str, tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The post-budget-pack `_backfill_hit_metadata` pass is the
    sole enrichment on the default (switch OFF) path; the early pre-pipeline
    pass runs only under the kill switch (legacy, verbatim). This fixture
    forces a genuine hit drop -- a schema/pattern-tagged hit gets pulled into
    `patterns_observed` by the concept-mode filter inside
    `_apply_post_rank_pipeline`, AFTER the early pass would have decoded it
    but BEFORE the final response is built -- so the differential actually
    exercises the "wasted decode avoided" claim, not just field-equality on
    an untrimmed list. The 4 enrichment fields must be byte-identical on
    every surviving hit AND anti_hit between switch ON and OFF.

    The graph node payload includes a real `created_at` (mirroring the
    production candidate-graph build at `core/__init__.py:766`) so the
    comparison is not confounded by `SimpleRecordView.created_at`'s
    `datetime.now()` fallback for a payload that omits the key.
    """
    _select_driver(driver, monkeypatch)
    store, graph, recs = _build_store_and_graph(
        tmp_path / f"store-{driver}", n=5, semantic_indices=[0],
    )
    for rec in recs:
        node = graph.get_payload(rec.id)
        node["created_at"] = str(rec.created_at)
        graph.set_node_payload(rec.id, node)
    node0 = graph.get_payload(recs[0].id)
    node0["tags"] = ["pattern:enrichment-parity-test"]
    graph.set_node_payload(recs[0].id, node0)

    # Store-only anti-hit target: never added to `graph`, so it can never be
    # a ranked hit candidate, but is still resolvable via store.get_batch()
    # once incident_edges surfaces it as a contradicts neighbour.
    anti_target = _make([0.0] * EMBED_DIM, text="anti-target", tier="episodic")
    store.insert(anti_target)
    store.boost_edges(
        [(recs[1].id, anti_target.id)], edge_type="contradicts", delta=1.0,
    )

    flush_record_buffer(store)
    flush_edge_buffer(store)
    store._build_exact_index_sync()
    assignment = _flat_assignment(recs)
    cue_vec = np.array(recs[0].embedding, dtype=np.float32)

    def _run(switch_off: bool):
        if switch_off:
            monkeypatch.setenv(CROSSING_KILL_SWITCH_ENV, "1")
        else:
            monkeypatch.delenv(CROSSING_KILL_SWITCH_ENV, raising=False)
        return recall_for_response(
            store=store, graph=graph, assignment=assignment, rich_club=[],
            embedder=_NullEmbedder(), cue="rec0 rec1 rec2 rec3 rec4",
            session_id="enrichment-parity-test", budget_tokens=2000,
            cue_embedding=cue_vec.tolist(), mode="concept",
        )

    resp_new = _run(switch_off=False)
    resp_legacy = _run(switch_off=True)

    # Fixture guarantees: the schema-tagged hit was dropped by the
    # concept-mode filter (budget-pack region, pipeline.py:2247-2297) on
    # both paths, and a real anti_hit was found on both paths.
    assert resp_new.patterns_observed, "fixture must force a schema/pattern drop"
    assert resp_legacy.patterns_observed, "fixture must force a schema/pattern drop"
    dropped_id = str(recs[0].id)
    assert dropped_id not in {str(h.record_id) for h in resp_new.hits}
    assert dropped_id not in {str(h.record_id) for h in resp_legacy.hits}
    assert resp_new.anti_hits, "fixture must guarantee a non-empty anti_hits set"
    assert resp_legacy.anti_hits, "fixture must guarantee a non-empty anti_hits set"

    new_hit_ids = {h.record_id for h in resp_new.hits}
    legacy_hit_ids = {h.record_id for h in resp_legacy.hits}
    assert new_hit_ids == legacy_hit_ids, (
        "switch ON/OFF must serve the identical surviving hit set (SC-5: "
        "no ranking/selection change)"
    )

    new_by_id = {h.record_id: h for h in resp_new.hits}
    legacy_by_id = {h.record_id: h for h in resp_legacy.hits}
    for rid, new_hit in new_by_id.items():
        legacy_hit = legacy_by_id[rid]
        assert _enrich_fields(new_hit) == _enrich_fields(legacy_hit), (
            f"hit {rid} enrichment fields diverged: "
            f"new={_enrich_fields(new_hit)} legacy={_enrich_fields(legacy_hit)}"
        )

    new_anti_ids = {h.record_id for h in resp_new.anti_hits}
    legacy_anti_ids = {h.record_id for h in resp_legacy.anti_hits}
    assert new_anti_ids == legacy_anti_ids

    new_anti_by_id = {h.record_id: h for h in resp_new.anti_hits}
    legacy_anti_by_id = {h.record_id: h for h in resp_legacy.anti_hits}
    for rid, new_hit in new_anti_by_id.items():
        legacy_hit = legacy_anti_by_id[rid]
        assert _enrich_fields(new_hit) == _enrich_fields(legacy_hit), (
            f"anti_hit {rid} enrichment fields diverged: "
            f"new={_enrich_fields(new_hit)} legacy={_enrich_fields(legacy_hit)}"
        )


def _add_edge_row(
    store, *, src: str, dst: str, edge_type: str = "contradicts", weight: float = 1.0,
) -> None:
    """Direct edges-table write bypassing UUID validation -- the established
    project fixture pattern for injecting a malformed row (test_pipeline_
    anti_hits_malformed.py's own helper), reused here so both files exercise
    the identical malformed-row shape.
    """
    tbl = store.db.open_table("edges")
    tbl.add([{
        "src": src,
        "dst": dst,
        "edge_type": edge_type,
        "weight": float(weight),
        "updated_at": datetime.now(timezone.utc),
    }])


def _make_hit(rid, surface: str = "primary topic") -> MemoryHit:
    return MemoryHit(
        record_id=rid,
        score=0.9,
        reason="test_hit",
        literal_surface=surface,
        adjacent_suggestions=[],
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_malformed_edge_warning_parity_across_kill_switch(
    driver: str, tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """`_find_anti_hits` derives the malformed-endpoint warning
    from the same rows its own contradicts query already fetches on the
    default (switch OFF) path, instead of a second `_log_malformed_anti_edges`
    query (legacy, switch ON, kept verbatim). Both a malformed dst and a
    malformed src row are present -- covering both branches
    `_log_malformed_anti_edges`'s row loop checks -- so the warning is
    emitted (or not) identically, and the resulting anti-hit set is
    unaffected, on both drivers.
    """
    _select_driver(driver, monkeypatch)
    store, graph, recs = _build_store_and_graph(tmp_path / f"store-{driver}", n=1)
    rid_hit = recs[0].id
    anti = _make([0.0] * EMBED_DIM, text="anti-target", tier="episodic")
    store.insert(anti)
    flush_record_buffer(store)

    _add_edge_row(store, src=str(rid_hit), dst=str(anti.id))
    _add_edge_row(store, src=str(rid_hit), dst="not-a-uuid")
    _add_edge_row(store, src="zzz-bad-src", dst=str(rid_hit))

    hit = _make_hit(rid_hit)
    graph_for_anti_hits = MemoryGraph()

    def _run(switch_off: bool):
        if switch_off:
            monkeypatch.setenv(CROSSING_KILL_SWITCH_ENV, "1")
        else:
            monkeypatch.delenv(CROSSING_KILL_SWITCH_ENV, raising=False)
        with caplog.at_level(logging.WARNING, logger="iai_mcp.pipeline"):
            result = _find_anti_hits(
                [hit], store, graph_for_anti_hits, k=3, records_cache=None,
            )
        messages = sorted(
            r.getMessage() for r in caplog.records
            if "anti_hits_skip_malformed_edge" in r.getMessage()
        )
        caplog.clear()
        return result, messages

    new_anti, new_messages = _run(switch_off=False)
    legacy_anti, legacy_messages = _run(switch_off=True)

    assert new_messages, "fixture must guarantee the malformed-edge warning fires"
    assert legacy_messages, "fixture must guarantee the malformed-edge warning fires"
    assert new_messages == legacy_messages, (
        f"malformed-edge warning diverged: new={new_messages} legacy={legacy_messages}"
    )

    assert {h.record_id for h in new_anti} == {h.record_id for h in legacy_anti}
    assert len(new_anti) == 1
    assert new_anti[0].record_id == anti.id


# ---------------------------------------------------------------------------
# Dispatch-level differentials for the auth-liveness single-borrow,
# pask_teachback reuse, and trajectory-coupling embedding-reuse
# consolidations -- these all live in iai_mcp.core.dispatch's memory_recall
# branch (above/around recall_for_response), so they exercise the full
# dispatch call rather than recall_for_response directly.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_authority_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IAI_MCP_EXACT_AUTHORITY_OFF", raising=False)


def _seeded_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _orthogonal_vec(seed: int) -> list[float]:
    """A unit vector with near-zero cosine similarity to _seeded_vec(seed)."""
    rng = np.random.default_rng(seed + 9000)
    v = rng.random(EMBED_DIM).astype(np.float32) - 0.5
    return (v / np.linalg.norm(v)).tolist()


def _make_rec(
    rid: UUID, seed: int, surface: str, *,
    embedding: "list[float] | None" = None,
    created_at: "datetime | None" = None,
) -> MemoryRecord:
    # A caller-supplied created_at keeps two independently-built stores
    # (one per switch state) from picking up a wall-clock skew of a few
    # milliseconds between construction calls -- that skew would otherwise
    # leak into the score's age term at full float precision even though it
    # rounds identically at 2 decimals, breaking byte-identity comparisons
    # that have nothing to do with the kill switch.
    now = created_at if created_at is not None else datetime.now(timezone.utc)
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


class _StubEmbedder:
    """Deterministic stand-in embedder -- a fixed cue vector regardless of text."""

    def __init__(self, vec: list[float]) -> None:
        self._vec = vec

    def embed(self, _text: str) -> list[float]:
        return list(self._vec)


def _stub_embedder_for_store(monkeypatch: pytest.MonkeyPatch, vec: list[float]) -> None:
    stub_embedder_for_store(monkeypatch, _StubEmbedder(vec))


_HIT_CMP_EXACT_FIELDS = ("record_id", "reason", "literal_surface", "epistemic_status", "salience_level")


def _assert_hit_parity(new_hit: dict, legacy_hit: dict, *, rid: str) -> None:
    """Exact equality on every qualitative field; a float-noise tolerance
    (not a kill-switch-related divergence) on score only -- two
    independently-built stores running the SAME exact-cosine scan can differ
    at the last couple of float32 ULPs from summation-order noise in the
    underlying dot product, well below any observable ranking difference.
    """
    for field in _HIT_CMP_EXACT_FIELDS:
        assert new_hit.get(field) == legacy_hit.get(field), (
            f"hit {rid} field {field!r} diverged: "
            f"new={new_hit.get(field)!r} legacy={legacy_hit.get(field)!r}"
        )
    new_score = float(new_hit.get("score", 0.0))
    legacy_score = float(legacy_hit.get("score", 0.0))
    assert math.isclose(new_score, legacy_score, rel_tol=1e-6, abs_tol=1e-6), (
        f"hit {rid} score diverged beyond float32 noise: "
        f"new={new_score} legacy={legacy_score}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_auth_liveness_single_borrow_parity_across_kill_switch(
    driver: str, tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The auth-liveness SELECT and the auth-new get_batch run
    inside ONE ro_conn() borrow (switch OFF, new) instead of two (switch ON,
    legacy verbatim). The fixture forces both halves of the merged call to
    actually fire: an ANN-tier miss so the authority path resolves a
    genuinely NEW candidate via get_batch(conn=...) reused across the single
    borrow, and a tombstoned decoy (bypassing invalidate_exact_index so the
    resident matrix stays stale) so the liveness SELECT inside that same
    borrow must actually filter something on both borrow shapes.
    """
    _select_driver(driver, monkeypatch)
    cue_vec = _seeded_vec(41)
    target_id = uuid4()
    victim_id = uuid4()
    filler_ids = [uuid4() for _ in range(3)]
    shared_created_at = datetime.now(timezone.utc)

    def _run(switch_off: bool) -> dict:
        # A fresh, independently-built store per switch state -- dispatch()
        # has real side effects (queue_reinforce, potentiate_coactivation)
        # that mutate degree/stability on the SAME store across sequential
        # calls, which would corrupt a same-store before/after comparison
        # with drift unrelated to the kill switch. Deterministic ids/
        # embeddings/timestamps make the two stores identical at recall time.
        store = MemoryStore(path=tmp_path / f"store-{driver}-{'legacy' if switch_off else 'new'}")
        target = _make_rec(
            target_id, seed=41, surface="auth single-borrow target",
            embedding=cue_vec, created_at=shared_created_at,
        )
        store.insert(target)
        victim = _make_rec(
            victim_id, seed=42, surface="soon to be tombstoned",
            embedding=cue_vec, created_at=shared_created_at,
        )
        store.insert(victim)
        for fid, seed in zip(filler_ids, range(300, 303)):
            store.insert(_make_rec(
                fid, seed=seed, surface=f"auth filler {seed}", created_at=shared_created_at,
            ))
        flush_record_buffer(store)

        _stub_embedder_for_store(monkeypatch, cue_vec)

        _orig_query_similar = store.query_similar

        def _query_similar_missing_targets(vec, *args, **kwargs):
            pairs = _orig_query_similar(vec, *args, **kwargs)
            return [
                (r, s) for r, s in pairs
                if getattr(r, "id", None) not in (target_id, victim_id)
            ]

        monkeypatch.setattr(store, "query_similar", _query_similar_missing_targets)

        store._build_exact_index_sync()
        warm_pairs = store.exact_top_k(cue_vec, k=10)
        assert any(rid == victim_id for rid, _ in warm_pairs), (
            "test setup: victim must be present in the warmed matrix before tombstoning"
        )

        # Tombstone bypassing invalidate_exact_index() -- the resident matrix
        # stays stale, so ONLY the liveness SELECT can catch this.
        with store.db._conn_lock:
            store.db._conn.execute(
                "UPDATE records SET tombstoned_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), str(victim_id)),
            )

        if switch_off:
            monkeypatch.setenv(CROSSING_KILL_SWITCH_ENV, "1")
        else:
            monkeypatch.delenv(CROSSING_KILL_SWITCH_ENV, raising=False)
        import iai_mcp.pipeline as _pm
        _pm._last_recall_latency_ms = 0.0
        return dispatch(store, "memory_recall", {
            "cue": "auth single-borrow parity probe",
            "session_id": "auth-single-borrow-parity",
            "budget_tokens": 2000,
            "cue_embedding": cue_vec,
        })

    resp_new = _run(switch_off=False)
    resp_legacy = _run(switch_off=True)

    for resp, label in ((resp_new, "new"), (resp_legacy, "legacy")):
        assert resp.get("exact_authority_used") is True, (
            f"fixture must exercise the authority merge on the {label} path; got {resp}"
        )
        hit_ids = {h["record_id"] for h in resp["hits"]}
        anti_ids = {h["record_id"] for h in resp.get("anti_hits", [])}
        assert str(target_id) in hit_ids, f"{label}: authority target must surface; hits={hit_ids}"
        assert str(victim_id) not in hit_ids, (
            f"{label}: tombstoned victim must never surface as a hit; hits={hit_ids}"
        )
        assert str(victim_id) not in anti_ids, (
            f"{label}: tombstoned victim must never surface as an anti_hit; anti_hits={anti_ids}"
        )

    new_hit_ids = {h["record_id"] for h in resp_new["hits"]}
    legacy_hit_ids = {h["record_id"] for h in resp_legacy["hits"]}
    assert new_hit_ids == legacy_hit_ids, (
        "switch ON/OFF must merge the identical candidate set (SC-3 #1: merged "
        f"_candidate_recs + _live_auth_ids byte-identical); new={new_hit_ids} "
        f"legacy={legacy_hit_ids}"
    )

    new_by_id = {h["record_id"]: h for h in resp_new["hits"]}
    legacy_by_id = {h["record_id"]: h for h in resp_legacy["hits"]}
    for rid, new_hit in new_by_id.items():
        _assert_hit_parity(new_hit, legacy_by_id[rid], rid=rid)


# ---------------------------------------------------------------------------
# pask_teachback reuse of _paired_edges with a candidate-membership guard.
#
# Empirically probed before writing these fixtures: escalate_recall_
# candidates (pipeline.py) restricts its widen to
# `set(graph.iter_nodes())` (pipeline.py:1284), and graph nodes are added
# ONLY by core/__init__.py from `_candidate_recs` before recall_for_response
# is ever called -- pipeline.py never calls graph.add_node itself. A probe
# script confirmed a widened id absent from the initial candidate pool never
# survives into resp.hits (it is filtered out before merging into
# records_cache). The membership guard therefore cannot currently be forced
# false via escalation; the reliably-constructible trigger for
# _all_cand_ids/_contr_edges being unavailable is the cortex-fallback /
# records_count==0 path, which reaches the pask block with those two
# variables at their pre-declared None default. See deferred-items.md.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_pask_teachback_reuse_high_confidence_parity_across_kill_switch(
    driver: str, tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pask_teachback reuse arm: two candidate hits with a contradicts edge
    between them, both members of _all_cand_ids -- the membership guard
    holds, so switch OFF (new) passes _contr_edges into verify_hit_set
    instead of it issuing its own query (switch ON, legacy verbatim).

    The edge is stored in the OPPOSITE direction from the reuse path's
    lexicographic canonicalization on purpose -- proving the direction
    divergence documented in pask_teachback._pairs_from_contradicts_edges is
    real and exercised, not dodged. Cardinality/boolean/pair-SET are
    asserted byte-identical; tuple direction is asserted per-path instead
    (the legacy raw SELECT has no ORDER BY and is not a direction contract
    the reuse path can or should reproduce).
    """
    _select_driver(driver, monkeypatch)
    cue_vec = _seeded_vec(51)
    id_a = uuid4()
    id_b = uuid4()
    shared_created_at = datetime.now(timezone.utc)
    smaller_id, larger_id = sorted([id_a, id_b], key=str)

    def _run(switch_off: bool) -> dict:
        store = MemoryStore(
            path=tmp_path / f"store-{driver}-{'legacy' if switch_off else 'new'}-pask-reuse",
        )
        store.insert(_make_rec(
            id_a, seed=51, surface="pask reuse hit alpha",
            embedding=cue_vec, created_at=shared_created_at,
        ))
        store.insert(_make_rec(
            id_b, seed=52, surface="pask reuse hit beta",
            embedding=cue_vec, created_at=shared_created_at,
        ))
        flush_record_buffer(store)
        # Deliberately reversed: src=larger, dst=smaller -- the opposite of
        # the reuse path's canonical (smaller, larger) sort order.
        store.add_contradicts_edge(larger_id, smaller_id)
        flush_edge_buffer(store)

        _stub_embedder_for_store(monkeypatch, cue_vec)
        monkeypatch.delenv("IAI_MCP_EXACT_AUTHORITY_OFF", raising=False)
        if switch_off:
            monkeypatch.setenv(CROSSING_KILL_SWITCH_ENV, "1")
        else:
            monkeypatch.delenv(CROSSING_KILL_SWITCH_ENV, raising=False)
        import iai_mcp.pipeline as _pm
        store._build_exact_index_sync()
        _pm._last_recall_latency_ms = 0.0
        return dispatch(store, "memory_recall", {
            "cue": "pask reuse parity probe",
            "session_id": "pask-reuse-parity",
            "budget_tokens": 2000,
            "cue_embedding": cue_vec,
        })

    resp_new = _run(switch_off=False)
    resp_legacy = _run(switch_off=True)

    tb_new = resp_new.get("pask_teachback")
    tb_legacy = resp_legacy.get("pask_teachback")
    assert tb_new is not None, f"fixture must produce a pask_teachback key; got {resp_new}"
    assert tb_legacy is not None, f"fixture must produce a pask_teachback key; got {resp_legacy}"

    assert tb_new["hit_count"] == 2, f"fixture must serve both hits; got {tb_new}"
    assert tb_legacy["hit_count"] == 2, f"fixture must serve both hits; got {tb_legacy}"
    assert tb_new["has_contradictions"] is True
    assert tb_legacy["has_contradictions"] is True
    assert len(tb_new["contradiction_pairs"]) == 1
    assert len(tb_legacy["contradiction_pairs"]) == 1

    new_pair_sets = {frozenset(p) for p in tb_new["contradiction_pairs"]}
    legacy_pair_sets = {frozenset(p) for p in tb_legacy["contradiction_pairs"]}
    expected = {frozenset((str(id_a), str(id_b)))}
    assert new_pair_sets == expected, f"new path pair set diverged: {new_pair_sets}"
    assert legacy_pair_sets == expected, f"legacy path pair set diverged: {legacy_pair_sets}"

    # Positive assertion: the reuse (new) path canonicalizes to lexicographic
    # order, independent of storage direction.
    assert tb_new["contradiction_pairs"][0] == (str(smaller_id), str(larger_id))
    # The legacy (raw SQL) path reports the TRUE stored direction, proving
    # the deliberately-reversed fixture actually diverges rather than
    # coincidentally matching the canonical order.
    assert tb_legacy["contradiction_pairs"][0] == (str(larger_id), str(smaller_id))


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_pask_teachback_fallback_when_candidate_set_unavailable(
    driver: str, tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pask_teachback fallback arm: the cortex-fallback recall path (daemon
    reports SLEEP) never computes _all_cand_ids/_contr_edges -- they stay at
    their pre-declared None default. The membership guard must route to
    verify_hit_set's own exact query on this path (rather than raising
    NameError or reusing stale/absent candidate data) on BOTH switch states,
    since the crossing-consolidation switch only selects between two
    _candidate_recs-dependent branches that never run on the cortex-fallback
    path at all -- both switch states hit the identical own-query code
    here, so byte-identity is guaranteed by construction, and this test
    proves the guard resolves cleanly rather than crashing.
    """
    _select_driver(driver, monkeypatch)
    cue_vec = _seeded_vec(61)
    id_a = uuid4()
    id_b = uuid4()
    shared_created_at = datetime.now(timezone.utc)

    def _run(switch_off: bool) -> dict:
        store = MemoryStore(
            path=tmp_path / f"store-{driver}-{'legacy' if switch_off else 'new'}-pask-fallback",
        )
        store.insert(_make_rec(
            id_a, seed=61, surface="pask fallback hit alpha",
            embedding=cue_vec, created_at=shared_created_at,
        ))
        store.insert(_make_rec(
            id_b, seed=62, surface="pask fallback hit beta",
            embedding=cue_vec, created_at=shared_created_at,
        ))
        flush_record_buffer(store)
        store.add_contradicts_edge(id_a, id_b)
        flush_edge_buffer(store)

        _stub_embedder_for_store(monkeypatch, cue_vec)
        monkeypatch.setenv("IAI_MCP_EXACT_AUTHORITY_OFF", "1")

        import iai_mcp.daemon_state as _dstate
        monkeypatch.setattr(_dstate, "load_state", lambda: {"current_state": "SLEEP"})

        if switch_off:
            monkeypatch.setenv(CROSSING_KILL_SWITCH_ENV, "1")
        else:
            monkeypatch.delenv(CROSSING_KILL_SWITCH_ENV, raising=False)
        import iai_mcp.pipeline as _pm
        store._build_exact_index_sync()
        _pm._last_recall_latency_ms = 0.0
        return dispatch(store, "memory_recall", {
            "cue": "pask fallback parity probe",
            "session_id": "pask-fallback-parity",
            "budget_tokens": 2000,
            "cue_embedding": cue_vec,
        })

    resp_new = _run(switch_off=False)
    resp_legacy = _run(switch_off=True)

    assert resp_new.get("_source") == "cortex-fallback", f"fixture must exercise cortex-fallback; got {resp_new}"
    assert resp_legacy.get("_source") == "cortex-fallback", f"fixture must exercise cortex-fallback; got {resp_legacy}"

    tb_new = resp_new.get("pask_teachback")
    tb_legacy = resp_legacy.get("pask_teachback")
    assert tb_new is not None, f"fixture must produce a pask_teachback key; got {resp_new}"
    assert tb_legacy is not None, f"fixture must produce a pask_teachback key; got {resp_legacy}"
    assert tb_new["hit_count"] >= 2, f"fixture must serve both hits; got {tb_new}"
    assert tb_new == tb_legacy, (
        "cortex-fallback pask response must be identical across switch states "
        f"(identical own-query code path either way): new={tb_new} legacy={tb_legacy}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_pask_teachback_guard_logic_direct_reuse_vs_query_parity(
    driver: str, tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pask_teachback direct guard-logic arm: verify_hit_set's own-query path
    (contradicts_edges=None) and its reuse path (contradicts_edges passed,
    hand-built in the exact shape store.incident_edges(..., neighbor_keys_
    as_str=True) produces) must agree on cardinality/boolean/pair-SET for
    the same stored edge -- including a deliberately reversed-direction edge
    so the canonicalization divergence documented above is asserted here
    too, independent of the full dispatch path.
    """
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path / f"store-{driver}-pask-guard-direct")
    id_a, id_b, id_c = uuid4(), uuid4(), uuid4()
    smaller_id, larger_id = sorted([id_a, id_b], key=str)

    for i, (rid, surface) in enumerate(((id_a, "alpha"), (id_b, "beta"), (id_c, "gamma"))):
        store.insert(_make_rec(rid, seed=71 + i, surface=surface))
    flush_record_buffer(store)
    store.add_contradicts_edge(larger_id, smaller_id)
    flush_edge_buffer(store)

    own_query_result = verify_hit_set(store, [id_a, id_b, id_c])

    # Hand-built incident-edges-shaped adjacency: symmetric (both directions
    # present), exactly as store.incident_edges(..., neighbor_keys_as_str=
    # True) would return for this single stored edge -- matching what
    # core/__init__.py's _contr_edges actually carries in production.
    contradicts_edges = {
        larger_id: [(str(smaller_id), "contradicts", 1.0)],
        smaller_id: [(str(larger_id), "contradicts", 1.0)],
        id_c: [],
    }
    reuse_result = verify_hit_set(
        store, [id_a, id_b, id_c], contradicts_edges=contradicts_edges,
    )

    assert own_query_result["has_contradictions"] is True
    assert reuse_result["has_contradictions"] is True
    assert own_query_result["hit_count"] == reuse_result["hit_count"] == 3
    assert len(own_query_result["contradiction_pairs"]) == 1
    assert len(reuse_result["contradiction_pairs"]) == 1

    expected = {frozenset((str(id_a), str(id_b)))}
    assert {frozenset(p) for p in own_query_result["contradiction_pairs"]} == expected
    assert {frozenset(p) for p in reuse_result["contradiction_pairs"]} == expected

    assert reuse_result["contradiction_pairs"][0] == (str(smaller_id), str(larger_id)), (
        "reuse path must canonicalize to lexicographic order"
    )
    assert own_query_result["contradiction_pairs"][0] == (str(larger_id), str(smaller_id)), (
        "own-query path must report the true stored direction (fixture is "
        "deliberately reversed relative to canonical order)"
    )

    # A hit_id entirely absent from contradicts_edges' keys (e.g. the
    # membership guard would have caught this upstream and fallen back to
    # None) must never raise -- it is simply excluded from pairs.
    partial_edges = {larger_id: [(str(smaller_id), "contradicts", 1.0)]}
    partial_result = verify_hit_set(
        store, [id_a, id_b], contradicts_edges=partial_edges,
    )
    assert isinstance(partial_result["has_contradictions"], bool)


# ---------------------------------------------------------------------------
# Trajectory-coupling top-5 hit embeddings read from
# _candidate_recs, get_batch only for ids absent from it.
#
# Same structural shape as the pask_teachback reuse guard above (see
# deferred-items.md item 2): a served hit currently cannot be absent from
# _candidate_recs via escalation, so the "escalating cue (fallback
# non-empty)" arm is not naturally constructible. The fallback (get_batch for
# ids not in _candidate_recs) is implemented defensively regardless -- the
# cortex-fallback path exercises it in full (every top-5 id is a miss,
# since _candidate_recs stays at its pre-declared None default there),
# which is a reliably-constructible trigger for the same branch. Measured
# firing rate on constructible fixtures: 0% partial-miss, 100% full-miss
# only on the cortex-fallback/records_count==0 paths -- a structural
# consequence of pipeline.py:1284, not a fixture gap.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_trajectory_coupling_candidate_recs_reuse_parity_across_kill_switch(
    driver: str, tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trajectory-coupling reuse arm: every top-5 hit id is present in
    _candidate_recs (the only reachable case in production today -- see the
    module docstring above this test). Switch OFF reads .embedding from
    _candidate_recs directly instead of the unconditional get_batch (switch
    ON, legacy verbatim); _last_injection_embedding/_last_injection_ids must
    match across switch states.

    Positive canary (switch OFF only): store.get_batch's returned .embedding
    is poisoned to an orthogonal sentinel for the trajectory ids, but ONLY
    on default-decode (`decode="full"`) calls -- every `_candidate_recs`-
    populating call in the dispatch path (hop expansion, auth-new) passes
    `decode="rank"` explicitly, so this never poisons `_candidate_recs`
    itself. Other default-decode fields (session_id, captured_at, ...) a
    call elsewhere (e.g. hit-metadata backfill) might legitimately need for
    these same ids stay untouched, so the poison can only leak into the
    result THROUGH the trajectory-coupling injection code specifically. If
    the reuse branch falls through to get_batch for any top-5 hit, the
    poisoned value measurably shifts emb_new toward the sentinel and the
    assertions below fail.
    """
    _select_driver(driver, monkeypatch)
    cue_vec = _seeded_vec(81)
    poison_vec = _orthogonal_vec(81)
    target_id = uuid4()
    filler_ids = [uuid4() for _ in range(2)]
    shared_created_at = datetime.now(timezone.utc)

    def _run(switch_off: bool, poison_get_batch: bool = False):
        store = MemoryStore(
            path=tmp_path / f"store-{driver}-{'legacy' if switch_off else 'new'}-traj-reuse",
        )
        store.insert(_make_rec(
            target_id, seed=81, surface="trajectory reuse target",
            embedding=cue_vec, created_at=shared_created_at,
        ))
        for fid, seed in zip(filler_ids, range(400, 402)):
            store.insert(_make_rec(
                fid, seed=seed, surface=f"trajectory filler {seed}",
                created_at=shared_created_at,
            ))
        flush_record_buffer(store)

        _stub_embedder_for_store(monkeypatch, cue_vec)
        monkeypatch.delenv("IAI_MCP_EXACT_AUTHORITY_OFF", raising=False)
        if switch_off:
            monkeypatch.setenv(CROSSING_KILL_SWITCH_ENV, "1")
        else:
            monkeypatch.delenv(CROSSING_KILL_SWITCH_ENV, raising=False)
        import iai_mcp.core as _core_mod
        import iai_mcp.pipeline as _pm
        store._build_exact_index_sync()
        _pm._last_recall_latency_ms = 0.0

        if poison_get_batch:
            _trajectory_ids = {target_id, *filler_ids}
            _real_get_batch = store.get_batch

            def _poisoned_get_batch(ids, **kwargs):
                result = _real_get_batch(ids, **kwargs)
                if kwargs.get("decode", "full") == "full":
                    for rid, rec in result.items():
                        if rid in _trajectory_ids:
                            rec.embedding = list(poison_vec)
                return result

            monkeypatch.setattr(store, "get_batch", _poisoned_get_batch)

        resp = dispatch(store, "memory_recall", {
            "cue": "trajectory reuse parity probe",
            "session_id": "trajectory-reuse-parity",
            "budget_tokens": 2000,
            "cue_embedding": cue_vec,
        })
        emb = _core_mod._last_injection_embedding
        return (
            resp,
            list(emb) if emb is not None else None,
            list(_core_mod._last_injection_ids),
        )

    resp_new, emb_new, ids_new = _run(switch_off=False, poison_get_batch=True)
    resp_legacy, emb_legacy, ids_legacy = _run(switch_off=True)

    assert resp_new["hits"], "fixture must serve at least one hit"
    assert resp_legacy["hits"], "fixture must serve at least one hit"
    assert ids_new, "fixture must populate _last_injection_ids"
    assert ids_legacy, "fixture must populate _last_injection_ids"
    assert ids_new == ids_legacy, (
        f"_last_injection_ids diverged: new={ids_new} legacy={ids_legacy}"
    )
    assert emb_new is not None, "fixture must populate _last_injection_embedding"
    assert emb_legacy is not None, "fixture must populate _last_injection_embedding"
    assert len(emb_new) == len(emb_legacy)

    # Positive proof the reuse branch fired: cosine similarity between the
    # new-path injection embedding and the poison sentinel must stay low.
    # A full fallthrough to the poisoned get_batch for every top-5 hit
    # collapses emb_new onto poison_vec (cos -> 1.0); a partial fallthrough
    # still measurably pulls cos upward. Legacy is unpoisoned and serves as
    # the natural baseline (its own poison-cosine is unrelated noise).
    _emb_new_arr = np.array(emb_new, dtype=np.float32)
    _emb_new_norm = float(np.linalg.norm(_emb_new_arr))
    assert _emb_new_norm > 0.0, "new-path injection embedding must be non-zero"
    _poison_cos_new = float(np.dot(_emb_new_arr / _emb_new_norm, poison_vec))
    assert _poison_cos_new < 0.5, (
        "new-path _last_injection_embedding looks poisoned by store.get_batch "
        "-- the reuse branch fell through to get_batch for at least one "
        f"top-5 hit instead of reading _candidate_recs (cos-with-poison="
        f"{_poison_cos_new}); emb_new={emb_new}"
    )

    for a, b in zip(emb_new, emb_legacy):
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9), (
            f"_last_injection_embedding diverged beyond float32 noise: "
            f"new={emb_new} legacy={emb_legacy}"
        )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_trajectory_coupling_fallback_when_candidate_recs_unavailable(
    driver: str, tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trajectory-coupling fallback arm: the cortex-fallback recall path never
    computes _candidate_recs -- it stays at its pre-declared None default,
    forcing every top-5 hit to be a get_batch miss on the new path (switch
    OFF), the same call shape the legacy path (switch ON) always makes
    unconditionally. Proves _candidate_recs is None resolves safely (no
    crash, no NameError) rather than reusing stale/absent candidate data,
    and _last_injection_embedding/_last_injection_ids match across switch
    states.
    """
    _select_driver(driver, monkeypatch)
    cue_vec = _seeded_vec(91)
    target_id = uuid4()
    shared_created_at = datetime.now(timezone.utc)

    def _run(switch_off: bool):
        store = MemoryStore(
            path=tmp_path / f"store-{driver}-{'legacy' if switch_off else 'new'}-traj-fallback",
        )
        store.insert(_make_rec(
            target_id, seed=91, surface="trajectory fallback target",
            embedding=cue_vec, created_at=shared_created_at,
        ))
        flush_record_buffer(store)

        _stub_embedder_for_store(monkeypatch, cue_vec)
        monkeypatch.setenv("IAI_MCP_EXACT_AUTHORITY_OFF", "1")

        import iai_mcp.daemon_state as _dstate
        monkeypatch.setattr(_dstate, "load_state", lambda: {"current_state": "SLEEP"})

        if switch_off:
            monkeypatch.setenv(CROSSING_KILL_SWITCH_ENV, "1")
        else:
            monkeypatch.delenv(CROSSING_KILL_SWITCH_ENV, raising=False)
        import iai_mcp.core as _core_mod
        import iai_mcp.pipeline as _pm
        store._build_exact_index_sync()
        _pm._last_recall_latency_ms = 0.0
        resp = dispatch(store, "memory_recall", {
            "cue": "trajectory fallback parity probe",
            "session_id": "trajectory-fallback-parity",
            "budget_tokens": 2000,
            "cue_embedding": cue_vec,
        })
        emb = _core_mod._last_injection_embedding
        return (
            resp,
            list(emb) if emb is not None else None,
            list(_core_mod._last_injection_ids),
        )

    resp_new, emb_new, ids_new = _run(switch_off=False)
    resp_legacy, emb_legacy, ids_legacy = _run(switch_off=True)

    assert resp_new.get("_source") == "cortex-fallback", f"fixture must exercise cortex-fallback; got {resp_new}"
    assert resp_legacy.get("_source") == "cortex-fallback", f"fixture must exercise cortex-fallback; got {resp_legacy}"
    assert resp_new["hits"], "fixture must serve at least one hit"
    assert resp_legacy["hits"], "fixture must serve at least one hit"
    assert ids_new == ids_legacy, (
        f"_last_injection_ids diverged: new={ids_new} legacy={ids_legacy}"
    )
    assert emb_new is not None, "fixture must populate _last_injection_embedding"
    assert emb_legacy is not None, "fixture must populate _last_injection_embedding"
    assert len(emb_new) == len(emb_legacy)
    for a, b in zip(emb_new, emb_legacy):
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9), (
            f"_last_injection_embedding diverged beyond float32 noise: "
            f"new={emb_new} legacy={emb_legacy}"
        )


# ---------------------------------------------------------------------------
# Permanent regression guard: fixture-store crossing-count reduces.
#
# Drives bench/crossing_count_probe.py's own counting harness (not a real-
# store bench measurement -- a small in-process fixture store) so a future
# edit that re-introduces a collapsed crossing (e.g. reverting the
# pask_teachback reuse) fails CI without requiring $IAI_ISO. The fixture is
# the pask_teachback reuse-vs-own-query pair (two candidate hits joined by a
# contradicts edge, both members of the frozen candidate set) -- the only
# wired consolidation whose trigger is reachable without the escalation-
# widen gap documented above the pask_teachback reuse tests.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_new_arm_synchronous_execute_count_below_legacy_arm_on_fixture_store(
    driver: str, tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Permanent crossing-count regression guard: the new arm (switch OFF)
    must issue strictly fewer engine crossings than the legacy arm (switch
    ON) on a small fixture store. This is a structural CI gate, not a bench
    measurement -- the real per-recall count only comes from
    bench/crossing_count_probe.py against $IAI_ISO.

    Guard metric: ``ro_conn_acq_main + total_main`` (RO acquisitions plus
    the probe's own "TOTAL synchronous engine execute()" figure), not
    ``total_main`` alone. ``iai_mcp.hippo._ro_pool``'s own module docstring
    states the RO connection pool is constructed ONLY for the lilli driver
    ("On the stdlib driver this module is not constructed at all"), so on
    stdlib every one of ``total_main``'s three components
    (``ro_execute_main``/``probe_execute_main``/``writer_execute_main``)
    stays at 0 for both switch states -- a vacuous 0<0 comparison that
    proves nothing. ``ro_conn_acq_main`` (patched directly on
    ``HippoDB.ro_conn``, independent of which internal branch serves the
    call) is the one counter this probe exposes that is meaningful on both
    drivers; empirically confirmed non-vacuous on both (measured this
    session: stdlib new=8/legacy=12, lilli new=47/legacy=59 for this
    fixture's finding-#3 trigger).

    Non-vacuity, both halves in one test: the count must differ (proving
    the consolidation actually fires) AND the served hit id set must be
    identical across arms (proving the count drop didn't come from serving
    fewer/different hits) -- carrying the full non-vacuity contract on one
    fixture instead of splitting it across two.
    """
    _select_driver(driver, monkeypatch)
    from bench.crossing_count_probe import _CrossingCounters, _install_patches

    cue_vec = _seeded_vec(101)
    id_a = uuid4()
    id_b = uuid4()
    shared_created_at = datetime.now(timezone.utc)

    def _measure(switch_off: bool) -> tuple[int, set]:
        store = MemoryStore(
            path=tmp_path / f"store-{driver}-{'legacy' if switch_off else 'new'}-guard",
        )
        store.insert(_make_rec(
            id_a, seed=101, surface="regression guard hit alpha",
            embedding=cue_vec, created_at=shared_created_at,
        ))
        store.insert(_make_rec(
            id_b, seed=102, surface="regression guard hit beta",
            embedding=cue_vec, created_at=shared_created_at,
        ))
        flush_record_buffer(store)
        store.add_contradicts_edge(id_a, id_b)
        flush_edge_buffer(store)

        _stub_embedder_for_store(monkeypatch, cue_vec)
        monkeypatch.delenv("IAI_MCP_EXACT_AUTHORITY_OFF", raising=False)
        if switch_off:
            monkeypatch.setenv(CROSSING_KILL_SWITCH_ENV, "1")
        else:
            monkeypatch.delenv(CROSSING_KILL_SWITCH_ENV, raising=False)
        import iai_mcp.pipeline as _pm
        store._build_exact_index_sync()
        _pm._last_recall_latency_ms = 0.0

        counters = _CrossingCounters()
        restore = _install_patches(counters)
        try:
            counters.in_dispatch = True
            try:
                resp = dispatch(store, "memory_recall", {
                    "cue": "regression guard probe",
                    "session_id": "crossing-guard",
                    "budget_tokens": 2000,
                    "cue_embedding": cue_vec,
                })
            finally:
                counters.in_dispatch = False
        finally:
            restore()
        hit_ids = {h["record_id"] for h in resp["hits"]}
        return counters.ro_conn_acq_main + counters.total_main, hit_ids

    new_total, new_hit_ids = _measure(switch_off=False)
    legacy_total, legacy_hit_ids = _measure(switch_off=True)
    monkeypatch.delenv(CROSSING_KILL_SWITCH_ENV, raising=False)

    assert new_hit_ids, "fixture must serve at least one hit"
    assert new_hit_ids == legacy_hit_ids, (
        "new-arm and legacy-arm must serve the identical hit set (the count "
        f"drop must come from fewer crossings, not fewer/different hits): "
        f"new={new_hit_ids} legacy={legacy_hit_ids}"
    )
    assert new_total < legacy_total, (
        "new-arm (switch OFF) engine-crossing count must be strictly below "
        f"legacy-arm (switch ON): new={new_total} legacy={legacy_total}"
    )
