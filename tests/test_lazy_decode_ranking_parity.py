"""Parity harness for the eager-vs-lazy candidate-decode ordering invariant.

There are two decode paths: the eager, full ``_from_row`` decode, and the
cheap ``RankCandidateView`` decode (``_from_row_rank_view``) used by
candidate-gathering call sites. ``IAI_MCP_LAZY_DECODE_OFF=1`` forces the
eager path so both paths can be measured on the same store/graph within one
process, following the project's existing bench-kill-switch convention
(``IAI_MCP_ANN_FAST_DECODE_OFF``).

``_recall_core``/``recall_for_response`` do not exercise the rank-view
decode tier at all on the default path. This file's coverage of that decode
tier lives entirely in two other places, both still live and exercised here: the
direct per-field decode comparison (``test_per_field_decode_level_completeness``,
``test_rank_view_language_legacy_schema_v1_parity``,
``test_per_field_score_level_load_bearing``,
``test_per_field_score_level_detection_not_reachable``), and the
dispatch-site tests, which exercise ``core.dispatch``'s own
``query_similar(decode="rank")``/``get_batch(decode="rank")`` call sites
(the rank-builder ephemeral-graph construction) directly.

Eager-vs-lazy comparisons tolerate a small absolute `score` delta
(``_SCORE_TOLERANCE``): a pre-existing ~1e-8-relative-magnitude float
non-determinism exists in the scoring pipeline, reproducible even between
two consecutive EAGER runs with the decode tier held fixed (unrelated to
this file's mechanism; hit ordering is unaffected). Every other
``MemoryHit`` field, and hit order itself, must stay byte-identical.

Dual-driver: parametrizes ``LILLI_STORAGE_DRIVER``.
"""
from __future__ import annotations

import collections
import inspect
import os
import re
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp import core as core_mod
from iai_mcp.community import CommunityAssignment
from iai_mcp.graph import MemoryGraph
from iai_mcp.pipeline import recall_for_response
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.store._store import MemoryStore as _MS
from iai_mcp.types import EMBED_DIM, STRUCTURE_HV_BYTES, MemoryRecord
from tests._helpers import stub_embedder_for_store
from tests.test_recall_core_unit import _build_store_and_graph, _flat_assignment

# Kill-switch this phase's mechanism plan will wire; documented here so the
# parity assertions below have a stable name to skip-guard against.
LAZY_DECODE_KILL_SWITCH_ENV = "IAI_MCP_LAZY_DECODE_OFF"

# Locked rank-view column set (subset of _RECORD_COLS) -- see
# 255-SCORE-FIELD-LIST.md section 5. Every field the score/reason formula
# reads pre-rank; the per-field non-vacuity control below parametrizes over
# this exact list once the mechanism exists.
RANK_VIEW_COLUMNS = (
    "id", "embedding", "literal_surface", "aaak_index", "created_at",
    "community_id", "structure_hv", "stability", "tags", "tier",
    "salience_level", "valence",
)


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


@pytest.fixture(autouse=True)
def _lex_fusion_off(monkeypatch: pytest.MonkeyPatch) -> None:
    # Defeats the cold-index/warm-index differential: lexical_query_warm
    # answers only from an already-built, generation-current index. Without
    # this, a cold-first/warm-second call pair produces a false differential
    # that has nothing to do with the decode-path change under test.
    monkeypatch.setenv("IAI_MCP_LEX_FUSION_OFF", "true")


@pytest.fixture
def _frozen_age_penalty(monkeypatch: pytest.MonkeyPatch):
    """Pin _age_penalty's wall-clock read to one instant for the duration of
    the fixture. _age_penalty calls datetime.now() fresh on every scoring
    pass (by design -- recency decay), so two recall_for_response calls at
    different real instants never score bit-identically on the age term
    regardless of decode tier. Freezing only this one function isolates the
    comparison to the decode-tier change under test."""
    from iai_mcp import pipeline as _pipeline_mod

    frozen_now = datetime.now(timezone.utc)

    def _frozen(created_at: datetime) -> float:
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        days = (frozen_now - created_at).total_seconds() / 86400.0
        if days < 0:
            return 0.0
        return min(1.0, days / _pipeline_mod.AGE_HALF_LIFE_DAYS)

    monkeypatch.setattr(_pipeline_mod, "_age_penalty", _frozen)


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
    """Never called: every recall in this harness supplies cue_embedding
    directly, so text->vector embedding is bypassed entirely."""

    DIM = EMBED_DIM

    def embed(self, text: str) -> list[float]:
        raise AssertionError("embedder.embed() called -- cue_embedding should have bypassed it")


def _seeded_unit_vec(seed: int, dim: int = EMBED_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_rec(
    vec: list[float], text: str, salience: str = "unflagged",
    session_id: str | None = None,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(), tier="episodic", literal_surface=text, aaak_index="",
        embedding=vec, community_id=None, centrality=0.0, detail_level=2,
        pinned=False, stability=0.0, difficulty=0.0, last_reviewed=None,
        never_decay=False, never_merge=False,
        provenance=([{"session_id": session_id}] if session_id else []),
        created_at=now, updated_at=now, tags=[], language="en",
        salience_level=salience,
    )


def _add_to_graph(graph: MemoryGraph, rec: MemoryRecord) -> None:
    graph.add_node(rec.id, community_id=None, embedding=list(rec.embedding))
    graph.set_node_payload(rec.id, {
        "embedding": list(rec.embedding), "surface": rec.literal_surface,
        "centrality": 0.0, "tier": rec.tier, "tags": rec.tags, "language": "en",
        "created_at": str(getattr(rec, "created_at", "") or ""),
    })


class _CueMix:
    """The four load-bearing cue fixtures, layered onto a base pool built
    via _build_store_and_graph so the driver x size parametrize scales a
    genuine corpus, not a fixed handful of records."""

    def __init__(self, store: MemoryStore, graph: MemoryGraph, recs: list[MemoryRecord]) -> None:
        self.store = store
        self.graph = graph
        self.recs = recs
        self.low_conf_vec: np.ndarray | None = None
        self.trigram_vec: np.ndarray | None = None
        self.trigram_cue_text = ""
        self.trigram_target_id: UUID | None = None
        self.salience_cue_vec: np.ndarray | None = None
        self.salience_target_id: UUID | None = None

    def build(self) -> None:
        # Low-confidence: a fresh random direction with no near-match in the
        # pool (the one-hot base pool and the cluster above are both ~
        # orthogonal to a distinct random high-dim seed).
        self.low_conf_vec = _seeded_unit_vec(9001)

        # Trigram/fts boost: a low-cosine record (orthogonal-ish embedding)
        # whose literal_surface trigram-matches the cue TEXT -- the cosine
        # alone would not surface it; the x2.0/x3.0 boost must. The cue
        # vector and the record's embedding are deliberately DIFFERENT seeds
        # (near-orthogonal) so cosine stays low; only the boost surfaces it.
        self.trigram_cue_text = "the quick brown fox jumps over the lazy dog"
        self.trigram_vec = _seeded_unit_vec(7500)
        trigram_rec = _make_rec(
            _seeded_unit_vec(7001).tolist(),
            "the quick brown fox jumps over a very lazy dog indeed",
        )
        self.store.insert(trigram_rec)
        self.recs.append(trigram_rec)
        self.trigram_target_id = trigram_rec.id
        _add_to_graph(self.graph, trigram_rec)

        # Salience-flagged record.
        self.salience_cue_vec = self.low_conf_vec
        salience_vec = _seeded_unit_vec(8001)
        salience_rec = _make_rec(
            salience_vec.tolist(), "critically salient record", salience="critical",
        )
        self.store.insert(salience_rec)
        self.recs.append(salience_rec)
        self.salience_target_id = salience_rec.id
        _add_to_graph(self.graph, salience_rec)

        flush_record_buffer(self.store)
        self.store._build_exact_index_sync()


@pytest.fixture
def _cue_mix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _make(driver: str, n: int) -> _CueMix:
        _select_driver(driver, monkeypatch)
        store, graph, recs = _build_store_and_graph(tmp_path / f"store-{driver}-{n}", n=n)
        mix = _CueMix(store, graph, recs)
        mix.build()
        return mix
    return _make


def _recall(
    mix: _CueMix, cue_text: str, cue_vec: np.ndarray,
    assignment: "CommunityAssignment | None" = None,
) -> object:
    # _flat_assignment mints a fresh random community_id per call; callers
    # comparing MemoryHit equality across repeated calls on the same mix
    # MUST pass one fixed assignment, or community_id alone breaks equality
    # for reasons unrelated to whatever the calls are actually comparing.
    if assignment is None:
        assignment = _flat_assignment(mix.recs)
    return recall_for_response(
        store=mix.store, graph=mix.graph, assignment=assignment, rich_club=[],
        embedder=_NullEmbedder(), cue=cue_text, session_id="parity-harness",
        budget_tokens=4000, cue_embedding=cue_vec.tolist(),
    )


# Absolute score tolerance for eager-vs-lazy / eager-vs-eager comparisons --
# absorbs the pre-existing ~1e-8-relative-magnitude float non-determinism in
# recall_for_response's scoring pipeline (see module docstring), never
# decode-tier drift: every non-score MemoryHit field, and hit order, stay
# asserted byte-identical.
_SCORE_TOLERANCE = 1e-6


def _assert_responses_match_within_score_tolerance(a: "object", b: "object", *, label: str) -> None:
    a_ids = [h.record_id for h in a.hits]
    b_ids = [h.record_id for h in b.hits]
    assert a_ids == b_ids, f"{label}: hit ordering diverged: {a_ids} vs {b_ids}"
    for h1, h2 in zip(a.hits, b.hits):
        assert h1.reason == h2.reason, (
            f"{label}: reason diverged for {h1.record_id}: {h1.reason!r} vs {h2.reason!r}"
        )
        assert h1.literal_surface == h2.literal_surface, (
            f"{label}: literal_surface diverged for {h1.record_id}"
        )
        assert h1.adjacent_suggestions == h2.adjacent_suggestions, (
            f"{label}: adjacent_suggestions diverged for {h1.record_id}"
        )
        assert h1.valid_from == h2.valid_from, f"{label}: valid_from diverged for {h1.record_id}"
        assert h1.valid_to == h2.valid_to, f"{label}: valid_to diverged for {h1.record_id}"
        assert h1.session_id == h2.session_id, f"{label}: session_id diverged for {h1.record_id}"
        assert h1.captured_at == h2.captured_at, f"{label}: captured_at diverged for {h1.record_id}"
        assert h1.community_id == h2.community_id, f"{label}: community_id diverged for {h1.record_id}"
        assert h1.epistemic_status == h2.epistemic_status, (
            f"{label}: epistemic_status diverged for {h1.record_id}"
        )
        assert h1.salience_level == h2.salience_level, (
            f"{label}: salience_level diverged for {h1.record_id}"
        )
        assert abs(h1.score - h2.score) <= _SCORE_TOLERANCE, (
            f"{label}: score diverged beyond the pre-existing-float-jitter "
            f"tolerance for {h1.record_id}: {h1.score!r} vs {h2.score!r}"
        )
    a_anti_ids = [h.record_id for h in a.anti_hits]
    b_anti_ids = [h.record_id for h in b.anti_hits]
    assert a_anti_ids == b_anti_ids, f"{label}: anti_hit ordering diverged"


@pytest.fixture
def _from_row_call_counter(monkeypatch: pytest.MonkeyPatch):
    """The same caller-tracking technique used to pin the decode split:
    records the immediate caller of every _from_row call,
    split by BUILD vs RECALL phase, so a future full-decode-call-count-drop
    assertion can diff eager vs lazy without a second measurement pass."""
    counts_build: collections.Counter = collections.Counter()
    counts_recall: collections.Counter = collections.Counter()
    phase = ["build"]

    orig = _MS._from_row

    def patched(self, row):
        caller = traceback.extract_stack()[-2]
        key = f"{caller.filename.split('/')[-1]}:{caller.lineno}:{caller.name}"
        (counts_build if phase[0] == "build" else counts_recall)[key] += 1
        return orig(self, row)

    monkeypatch.setattr(_MS, "_from_row", patched)

    class _Counter:
        def start_recall_phase(self) -> None:
            phase[0] = "recall"
            counts_recall.clear()

        @property
        def recall_total(self) -> int:
            return sum(counts_recall.values())

        @property
        def recall_breakdown(self) -> dict:
            return dict(counts_recall)

    return _Counter()


def _onehot_vec(idx: int, dim: int = EMBED_DIM) -> list[float]:
    v = [0.0] * dim
    v[idx] = 1.0
    return v


def _build_anti_hit_overlap_store(
    driver: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[MemoryStore, "object", "object", list, UUID, UUID]:
    """An anchor hit and a contradicts-edge neighbour that is ALSO a graph
    node -- already present in records_cache as a
    SimpleRecordView (provenance=[]) by the time _find_anti_hits runs,
    regardless of decode tier or escalation. Mirrors
    test_epistemic_status_recall_render.py's proven
    test_recall_for_response_anti_hit_from_graph_cache_carries_epistemic_status
    fixture shape (real build_runtime_graph, exactly-orthogonal onehot
    embeddings, a filler pool sized so relevance ranking -- not budget
    headroom -- excludes the contradicting neighbour from hits): a flat
    hand-built graph or a too-small filler pool lets every record clear
    the generous token budget regardless of relevance, which would make
    the anti-hit path trivially unreachable.
    """
    from iai_mcp import retrieve

    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path / f"anti-overlap-{driver}")

    fillers = []
    for i in range(60):
        f = _make_rec(_seeded_unit_vec(9000 + i).tolist(), f"anti-overlap filler {i}")
        store.insert(f)
        fillers.append(f)

    anchor = _make_rec(_onehot_vec(0), "alice's release ships Tuesday", session_id="sess-anchor")
    contra = _make_rec(
        _onehot_vec(1), "alice's release actually slipped", session_id="sess-contra",
    )
    store.insert(anchor)
    store.insert(contra)
    flush_record_buffer(store)

    tbl = store.db.open_table("edges")
    edge_rows = [{
        "src": str(anchor.id), "dst": str(contra.id), "edge_type": "contradicts",
        "weight": 1.0, "updated_at": datetime.now(timezone.utc),
    }]
    for j in range(0, len(fillers) - 1, 2):
        edge_rows.append({
            "src": str(fillers[j].id), "dst": str(fillers[j + 1].id),
            "edge_type": "hebbian", "weight": 1.0,
            "updated_at": datetime.now(timezone.utc),
        })
    tbl.add(edge_rows)

    g, a, rc = retrieve.build_runtime_graph(store)
    return store, g, a, rc, anchor.id, contra.id


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_anti_hit_overlap_session_id_captured_at_backfilled(
    driver, tmp_path, monkeypatch, _frozen_age_penalty,
):
    """An anti-hit id that is ALSO a graph node sits in
    records_cache as a SimpleRecordView (provenance=[]) by the time
    _find_anti_hits runs -- unlike hits, anti_hits are minted later, inside
    _apply_post_rank_pipeline, and never pass through recall_for_response's
    own _enrich_ids/_enrich_batch pass. record_id/score/reason equality
    would NOT catch this gap: both are identical on the broken code too.
    Only session_id/captured_at equality -- and non-None-ness -- catch it.
    """
    store, g, a, rc, anchor_id, contra_id = _build_anti_hit_overlap_store(
        driver, tmp_path, monkeypatch,
    )
    cue_vec = _onehot_vec(0)

    monkeypatch.setenv(LAZY_DECODE_KILL_SWITCH_ENV, "1")
    eager = recall_for_response(
        store=store, graph=g, assignment=a, rich_club=rc,
        embedder=_NullEmbedder(), cue="ignored -- a valid vector is supplied directly",
        session_id="parity-harness-antihit", budget_tokens=2000, mode="concept",
        cue_embedding=cue_vec,
    )

    monkeypatch.delenv(LAZY_DECODE_KILL_SWITCH_ENV, raising=False)
    lazy = recall_for_response(
        store=store, graph=g, assignment=a, rich_club=rc,
        embedder=_NullEmbedder(), cue="ignored -- a valid vector is supplied directly",
        session_id="parity-harness-antihit", budget_tokens=2000, mode="concept",
        cue_embedding=cue_vec,
    )

    eager_hit_ids = {h.record_id for h in eager.hits}
    assert anchor_id in eager_hit_ids, (
        "fixture precondition: anchor must rank as a hit for its "
        "contradicts edge to be looked up by _find_anti_hits at all"
    )
    assert contra_id not in eager_hit_ids, (
        "fixture precondition: contradicting must stay OUT of hits or the "
        "anti-hit path is never exercised"
    )

    eager_anti = [h for h in eager.anti_hits if h.record_id == contra_id]
    lazy_anti = [h for h in lazy.anti_hits if h.record_id == contra_id]
    assert eager_anti and lazy_anti, (
        "fixture precondition: contradicting neighbour must surface as an "
        f"anti-hit -- eager={eager.anti_hits!r} lazy={lazy.anti_hits!r}"
    )
    assert eager_anti[0].session_id == "sess-contra", (
        f"eager anti-hit session_id not backfilled: {eager_anti[0].session_id!r}"
    )
    assert eager_anti[0].captured_at is not None, (
        "eager anti-hit captured_at not backfilled"
    )
    assert lazy_anti[0] == eager_anti[0], (
        "lazy anti-hit diverges from eager on full MemoryHit equality "
        "(session_id/captured_at included)"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
@pytest.mark.parametrize("n", [12, 200, 1000])
def test_from_row_call_counter_hook_present(driver, n, _cue_mix, _from_row_call_counter):
    """Sanity: the caller-tracking counter fixture is wired and observes
    real _from_row activity during a recall -- the hook the mechanism and
    guards plans will diff eager-vs-lazy call counts against."""
    mix = _cue_mix(driver, n)
    _from_row_call_counter.start_recall_phase()

    _recall(mix, "counter sanity probe", mix.low_conf_vec)

    assert _from_row_call_counter.recall_total > 0, (
        "the _from_row call counter observed zero calls during a recall -- "
        "the counter hook is not wired correctly"
    )


# Fields the rank-view decoder is a strict decomposition of _from_row for,
# but which never reach _recall_core's scoring loop through records_cache --
# confirmed empirically: dispatch()'s ephemeral-graph payload builder
# (core/__init__.py, pinned by test_dispatch_candidate_payload_keys_unchanged)
# never propagates these three. Proven complete at the DECODE level
# (test_per_field_decode_level_completeness) instead of the score level.
_DECODE_ONLY_FIELDS = ("community_id", "structure_hv", "salience_level", "valence")

# Fields that flow rank-view -> dispatch()'s ephemeral-graph payload ->
# SimpleRecordView -> _recall_core's scoring loop -- provably load-bearing
# via an observable score/reason change (test_per_field_score_level_load_bearing).
_SCORE_LEVEL_FIELDS = (
    "embedding", "literal_surface", "aaak_index", "created_at",
    "stability", "tags", "tier",
)


def _rank_view_field_drop_sentinel(rv: "object", field: str) -> None:
    """Mutate `rv` (a RankCandidateView) to the value _from_row_rank_view
    would produce if it stopped reading `field`'s SQL column -- the class's
    own "not decoded" default for every field except created_at (whose
    default_factory produces a fresh, not fixed, sentinel)."""
    if field == "embedding":
        rv.embedding = []
    elif field == "literal_surface":
        rv.literal_surface = ""
    elif field == "aaak_index":
        rv.aaak_index = ""
    elif field == "created_at":
        rv.created_at = datetime.now(timezone.utc)
    elif field == "community_id":
        rv.community_id = None
    elif field == "structure_hv":
        rv.structure_hv = b""
    elif field == "stability":
        rv.stability = 0.0
    elif field == "tags":
        rv.tags = []
    elif field == "tier":
        rv.tier = "episodic"
    elif field == "salience_level":
        rv.salience_level = "unflagged"
    elif field == "valence":
        rv.valence = 0.0
    else:
        raise AssertionError(f"no drop sentinel defined for field {field!r}")


def _make_field_complete_rec() -> MemoryRecord:
    """A record whose value for every RANK_VIEW_COLUMNS field is distinct
    from that field's rank-view "not decoded" sentinel -- a drop that lands
    on the real value by coincidence would falsely pass the negative
    control, so every value here must differ from _rank_view_field_drop_sentinel's
    default."""
    now = datetime.now(timezone.utc) - timedelta(days=3)
    return MemoryRecord(
        id=uuid4(), tier="semantic", literal_surface="decode completeness probe surface",
        aaak_index="E:probeentity", embedding=_seeded_unit_vec(4001).tolist(),
        community_id=uuid4(), centrality=0.0, detail_level=2, pinned=False,
        stability=0.42, difficulty=0.0, last_reviewed=None, never_decay=False,
        never_merge=False, provenance=[{"session_id": "sess-decode"}],
        created_at=now, updated_at=now, tags=["tag-a", "tag-b"], language="en",
        structure_hv=b"\x01" * STRUCTURE_HV_BYTES, salience_level="critical",
        valence=0.73,
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
@pytest.mark.parametrize("field", [c for c in RANK_VIEW_COLUMNS if c != "id"])
def test_per_field_decode_level_completeness(driver, field, tmp_path, monkeypatch):
    """Per-field non-vacuity control, decode level: proves RankCandidateView
    is a strict per-field decomposition of _from_row for every locked
    column, and that removing a field from that decomposition is
    DETECTABLE, not silently absorbed.

    Positive control: the unmodified rank-view decode agrees with full
    decode on `field`, for a record whose value is distinct from the
    field's "not decoded" sentinel (_make_field_complete_rec) -- proves the
    field is normally decoded correctly.

    Negative control: with _from_row_rank_view patched to force `field` to
    its not-decoded sentinel, the rank-view decode now DISAGREES with full
    decode -- proves the comparison is sensitive to this field's absence,
    not a gate that could pass regardless of what the decoder does.
    """
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path / f"decode-complete-{driver}-{field}")
    rec = _make_field_complete_rec()
    store.insert(rec)
    flush_record_buffer(store)

    full = store.get_batch([rec.id])[rec.id]
    rank_baseline = store.get_batch([rec.id], decode="rank")[rec.id]

    def _val(obj: "object") -> "object":
        v = getattr(obj, field)
        return list(v) if isinstance(v, np.ndarray) else v

    assert _val(rank_baseline) == _val(full), (
        f"positive control failed: unmodified rank-view decode of {field!r} "
        f"({_val(rank_baseline)!r}) disagrees with full decode "
        f"({_val(full)!r}) -- the rank-view decoder is not byte-identical "
        f"for this field even before any drop is applied"
    )

    orig = _MS._from_row_rank_view

    def _patched(self, row, _field=field, _orig=orig):
        rv = _orig(self, row)
        _rank_view_field_drop_sentinel(rv, _field)
        return rv

    monkeypatch.setattr(_MS, "_from_row_rank_view", _patched)
    rank_dropped = store.get_batch([rec.id], decode="rank")[rec.id]

    assert _val(rank_dropped) != _val(full), (
        f"negative control failed: dropping {field!r} from the rank-view "
        f"decode did NOT change its value relative to full decode "
        f"({_val(rank_dropped)!r} == {_val(full)!r}) -- either the fixture's "
        f"real value coincides with the drop sentinel, or the field is not "
        f"actually part of the rank-view's decomposition"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_rank_view_language_legacy_schema_v1_parity(driver, tmp_path, monkeypatch):
    """Pins M-1 (255-REVIEW.md): `language` is not in RANK_VIEW_COLUMNS --
    it was added to RankCandidateView unilaterally, needed only because
    dispatch()'s ephemeral-graph payload builder reads `_rec.language`, and
    never went through the phase's locked-column process. That means
    test_per_field_decode_level_completeness does not parametrize over it,
    so nothing else in this file would catch a regression of
    _from_row_rank_view's legacy-schema-v1 language handling. A record with
    schema_version=1 and an empty language column must decode to the SAME
    value ("") on both tiers, mirroring _from_row's __LEGACY_EMPTY__
    special case verbatim -- before the fix, the rank-view silently
    produced "en" instead.
    """
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path / f"lang-legacy-{driver}")
    rec = _make_field_complete_rec()
    store.insert(rec)
    flush_record_buffer(store)

    tbl = store.db.open_table("records")
    tbl.update(where=f"id = '{rec.id}'", values={"language": "", "schema_version": 1})

    full = store.get_batch([rec.id], decode="full")[rec.id]
    rank = store.get_batch([rec.id], decode="rank")[rec.id]
    assert full.language == "", (
        "fixture precondition: full decode of a schema_version=1 "
        f"empty-language row must be '' (the __LEGACY_EMPTY__ unwrap) -- "
        f"got {full.language!r}"
    )
    assert rank.language == full.language, (
        f"get_batch: rank-view language diverges from full decode for a "
        f"legacy (schema_version=1, empty language) row: "
        f"rank={rank.language!r} full={full.language!r}"
    )

    qs_full = store.query_similar(rec.embedding, k=1, decode="full")
    qs_rank = store.query_similar(rec.embedding, k=1, decode="rank")
    assert qs_full[0][0].language == "" and qs_rank[0][0].language == "", (
        f"query_similar: rank-view language diverges from full decode on "
        f"the legacy row: rank={qs_rank[0][0].language!r} "
        f"full={qs_full[0][0].language!r}"
    )


def _build_score_level_field_store(
    driver: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str,
) -> tuple[MemoryStore, UUID, list[float], str]:
    """A target record + filler pool, sized like the dispatch-site fixture,
    with `field` set to a value that makes its own score/reason contribution
    observable -- the OTHER score-affecting fields stay at rank-view
    defaults so only `field`'s own drop is exercised (e.g. the tags case
    uses tier="episodic" so tier itself contributes nothing, isolating the
    xtier boost to the doc: tag alone; the tier case uses tags without a
    doc: tag so xtier is driven by tier="semantic" alone).
    """
    _select_driver(driver, monkeypatch)
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_DISPATCH_DIM))
    store = MemoryStore(path=tmp_path / f"field-detect-{driver}-{field}")

    cue_vec = _dispatch_seeded_vec(1)
    cue_text = "dispatch decode-tier parity probe alpha beta"
    now = datetime.now(timezone.utc)

    kwargs: dict = dict(
        id=uuid4(), tier="episodic", literal_surface=f"filler surface for {field}",
        aaak_index="", embedding=cue_vec.tolist(), community_id=None, centrality=0.0,
        detail_level=2, pinned=False, stability=0.5, difficulty=0.0,
        last_reviewed=None, never_decay=False, never_merge=False,
        provenance=[{"session_id": "sess-x"}], created_at=now, updated_at=now,
        tags=["capture"], language="en",
    )
    if field == "embedding":
        # A DIFFERENT record supplies the top-cosine match today; forcing
        # this record's embedding to the drop sentinel ([]) at decode time
        # collapses its cosine to 0 -- detectable via its own score/rank.
        pass
    elif field == "literal_surface":
        kwargs["literal_surface"] = cue_text
    elif field == "aaak_index":
        kwargs["aaak_index"] = "E:alpha"
    elif field == "created_at":
        kwargs["created_at"] = kwargs["updated_at"] = now - timedelta(days=30)
    elif field == "stability":
        kwargs["stability"] = 0.9
    elif field == "tags":
        kwargs["tags"] = ["doc:manual"]
    elif field == "tier":
        kwargs["tier"] = "semantic"
    else:
        raise AssertionError(f"no score-level fixture defined for field {field!r}")

    target = MemoryRecord(**kwargs)
    store.insert(target)
    for i in range(250):
        store.insert(_make_dispatch_rec(uuid4(), f"filler {i}", _dispatch_seeded_vec(2000 + i)))
    flush_record_buffer(store)
    store._build_exact_index_sync()
    return store, target.id, cue_vec.tolist(), cue_text


def _recall_via_hand_built_ephemeral_graph(
    store: MemoryStore, target_id: UUID, cue_vec: list[float], cue_text: str,
) -> "object":
    """Mirrors dispatch()'s own candidate-gathering + ephemeral-graph
    construction (core/__init__.py: query_similar(decode=...) -> graph.add_node
    + graph.set_node_payload with the SAME 9-key payload dispatch() writes)
    but calls recall_for_response directly instead of going through
    dispatch()'s authority-merge/exact-cosine wrapper -- that wrapper has a
    pre-existing centrality/ordering non-determinism on a fresh ephemeral
    graph, unrelated to decode tier, that would make a byte-exact score
    comparison here flaky for reasons having nothing to do with the field
    under test.
    """
    decode = "full" if os.environ.get(LAZY_DECODE_KILL_SWITCH_ENV) == "1" else "rank"
    pairs = store.query_similar(cue_vec, k=30, decode=decode)
    cand = {r.id: r for r, _s in pairs}
    if target_id not in cand:
        cand.update(store.get_batch([target_id], decode=decode))
    graph = MemoryGraph()
    for rec in cand.values():
        graph.add_node(
            rec.id, community_id=getattr(rec, "community_id", None),
            embedding=list(rec.embedding or []),
        )
        graph.set_node_payload(rec.id, {
            "embedding": list(rec.embedding or []),
            "surface": rec.literal_surface or "",
            "centrality": float(getattr(rec, "centrality", 0.0) or 0.0),
            "tier": rec.tier or "episodic",
            "tags": list(rec.tags or []),
            "language": rec.language or "en",
            "aaak_index": str(getattr(rec, "aaak_index", "") or ""),
            "created_at": str(getattr(rec, "created_at", "") or ""),
            "stability": float(getattr(rec, "stability", 0.5) or 0.5),
        })
    recs_list = list(cand.values())
    assignment = _flat_assignment(recs_list)
    return recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=[],
        embedder=_NullEmbedder(), cue=cue_text, session_id="field-detect",
        budget_tokens=3000, mode="concept", cue_embedding=cue_vec,
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
@pytest.mark.parametrize("field", _SCORE_LEVEL_FIELDS)
def test_per_field_score_level_load_bearing(
    driver, field, tmp_path, monkeypatch, _frozen_age_penalty,
):
    """Per-field non-vacuity control, score level, for the 7 fields that
    flow rank-view -> dispatch()'s ephemeral-graph payload -> scoring
    (_SCORE_LEVEL_FIELDS): dropping `field` from the rank-view decode must
    change the target's observable score/reason in recall_for_response's
    real scoring loop, proving the field is load-bearing for ranking, not
    merely present in the decoder's output tuple.
    """
    store, target_id, cue_vec, cue_text = _build_score_level_field_store(
        driver, tmp_path, monkeypatch, field,
    )

    monkeypatch.delenv(LAZY_DECODE_KILL_SWITCH_ENV, raising=False)
    baseline = _recall_via_hand_built_ephemeral_graph(store, target_id, cue_vec, cue_text)
    baseline_hit = next((h for h in baseline.hits if h.record_id == target_id), None)
    assert baseline_hit is not None, (
        f"fixture precondition ({field}): target record did not surface as "
        f"a hit at all -- baseline hits={[h.record_id for h in baseline.hits]}"
    )

    orig = _MS._from_row_rank_view

    def _patched(self, row, _field=field, _orig=orig):
        rv = _orig(self, row)
        if _field == "embedding":
            # _collect_graph_pool's own bounded fallback treats an EMPTY
            # graph-node embedding as "missing" and self-heals with a full
            # (non-rank) store.get_batch re-fetch -- the correct defensive
            # behavior in production, but it would silently repair the
            # standard "not decoded" ([]) sentinel before scoring ever sees
            # it, making the drop undetectable here for a reason that has
            # nothing to do with whether embedding is load-bearing. A
            # non-empty WRONG vector bypasses that fallback and reaches
            # scoring, which is what this control needs to prove.
            rv.embedding = _seeded_unit_vec(9999, dim=_DISPATCH_DIM).tolist()
        else:
            _rank_view_field_drop_sentinel(rv, _field)
        return rv

    monkeypatch.setattr(_MS, "_from_row_rank_view", _patched)
    dropped = _recall_via_hand_built_ephemeral_graph(store, target_id, cue_vec, cue_text)
    dropped_hit = next((h for h in dropped.hits if h.record_id == target_id), None)

    dropped_shape = (dropped_hit.score, dropped_hit.reason) if dropped_hit else None
    baseline_shape = (baseline_hit.score, baseline_hit.reason)
    assert dropped_shape != baseline_shape, (
        f"dropping {field!r} from the rank-view decode did not change the "
        f"target's score/reason -- baseline={baseline_shape!r} "
        f"dropped={dropped_shape!r} (field not load-bearing on this path, "
        f"or the fixture failed to isolate it)"
    )


def _build_decode_only_field_store(
    driver: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str,
) -> tuple[MemoryStore, UUID, list[float], str]:
    """Same shape as _build_score_level_field_store, for the 3 fields the
    dispatch-site ephemeral-graph payload never propagates -- the target
    carries a real, distinct, non-default value for `field` so a genuine
    score effect (if the field DID reach scoring) would be observable."""
    _select_driver(driver, monkeypatch)
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_DISPATCH_DIM))
    store = MemoryStore(path=tmp_path / f"decode-only-{driver}-{field}")

    cue_vec = _dispatch_seeded_vec(1)
    cue_text = "dispatch decode-tier parity probe alpha beta"
    now = datetime.now(timezone.utc)
    kwargs: dict = dict(
        id=uuid4(), tier="episodic", literal_surface="target for decode-only field",
        aaak_index="", embedding=cue_vec.tolist(), community_id=None, centrality=0.0,
        detail_level=2, pinned=False, stability=0.5, difficulty=0.0,
        last_reviewed=None, never_decay=False, never_merge=False,
        provenance=[{"session_id": "sess-x"}], created_at=now, updated_at=now,
        tags=["capture"], language="en",
    )
    if field == "community_id":
        kwargs["community_id"] = uuid4()
    elif field == "structure_hv":
        kwargs["structure_hv"] = b"\x01" * STRUCTURE_HV_BYTES
    elif field == "salience_level":
        kwargs["salience_level"] = "critical"
    else:
        raise AssertionError(f"no decode-only fixture defined for field {field!r}")

    target = MemoryRecord(**kwargs)
    store.insert(target)
    for i in range(250):
        store.insert(_make_dispatch_rec(uuid4(), f"filler {i}", _dispatch_seeded_vec(2000 + i)))
    flush_record_buffer(store)
    store._build_exact_index_sync()
    return store, target.id, cue_vec.tolist(), cue_text


@pytest.mark.xfail(
    reason=(
        "community_id/structure_hv/salience_level are correctly decoded by "
        "the rank-view (see test_per_field_decode_level_completeness) but "
        "never reach _recall_core's scoring loop: dispatch()'s "
        "ephemeral-graph payload builder does not propagate them (pinned by "
        "test_dispatch_candidate_payload_keys_unchanged's 9-key guard). "
        "structural_weight>0 makes the structure_hv scoring TERM live but "
        "the records_cache INPUT stays the class default (b\"\") on both "
        "decode tiers, so the drop is undetectable -- see docstring."
    ),
    strict=True,
)
@pytest.mark.parametrize("field", _DECODE_ONLY_FIELDS)
def test_per_field_score_level_detection_not_reachable(
    field, tmp_path, monkeypatch, _frozen_age_penalty,
):
    """Empirically confirms (not just asserts) that structural_weight>0-style
    score-level detection is NOT achievable today for
    community_id/structure_hv/salience_level: reactivate once a
    payload-builder change lets these fields reach records_cache -- an
    unexpected pass fails the suite, forcing the marker to be removed
    deliberately.
    """
    store, target_id, cue_vec, cue_text = _build_decode_only_field_store(
        driver="stdlib", tmp_path=tmp_path, monkeypatch=monkeypatch, field=field,
    )
    profile_state = {"structural_weight": 0.5} if field == "structure_hv" else None

    monkeypatch.delenv(LAZY_DECODE_KILL_SWITCH_ENV, raising=False)

    def _run() -> tuple:
        decode = "rank"
        pairs = store.query_similar(cue_vec, k=30, decode=decode)
        cand = {r.id: r for r, _s in pairs}
        if target_id not in cand:
            cand.update(store.get_batch([target_id], decode=decode))
        graph = MemoryGraph()
        for rec in cand.values():
            graph.add_node(
                rec.id, community_id=getattr(rec, "community_id", None),
                embedding=list(rec.embedding or []),
            )
            graph.set_node_payload(rec.id, {
                "embedding": list(rec.embedding or []),
                "surface": rec.literal_surface or "",
                "centrality": float(getattr(rec, "centrality", 0.0) or 0.0),
                "tier": rec.tier or "episodic",
                "tags": list(rec.tags or []),
                "language": rec.language or "en",
                "aaak_index": str(getattr(rec, "aaak_index", "") or ""),
                "created_at": str(getattr(rec, "created_at", "") or ""),
                "stability": float(getattr(rec, "stability", 0.5) or 0.5),
            })
        assignment = _flat_assignment(list(cand.values()))
        resp = recall_for_response(
            store=store, graph=graph, assignment=assignment, rich_club=[],
            embedder=_NullEmbedder(), cue=cue_text, session_id="decode-only-detect",
            budget_tokens=3000, mode="concept", cue_embedding=cue_vec,
            profile_state=profile_state,
        )
        hit = next((h for h in resp.hits if h.record_id == target_id), None)
        return (hit.score, hit.reason) if hit else None

    baseline_shape = _run()
    assert baseline_shape is not None, (
        f"fixture precondition ({field}): target did not surface as a hit"
    )

    orig = _MS._from_row_rank_view

    def _patched(self, row, _field=field, _orig=orig):
        rv = _orig(self, row)
        _rank_view_field_drop_sentinel(rv, _field)
        return rv

    monkeypatch.setattr(_MS, "_from_row_rank_view", _patched)
    dropped_shape = _run()

    assert dropped_shape != baseline_shape, (
        f"{field!r} IS now detectable at the score level "
        f"(baseline={baseline_shape!r} dropped={dropped_shape!r}) -- remove "
        f"this xfail and move {field!r} into _SCORE_LEVEL_FIELDS"
    )


_DISPATCH_DIM = 16
_DISPATCH_N = 250


@pytest.fixture(autouse=True)
def _small_embed_dim_for_dispatch(request, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.function.__name__.startswith("test_dispatch_"):
        monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_DISPATCH_DIM))


def _dispatch_seeded_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(_DISPATCH_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_dispatch_rec(rid: UUID, surface: str, emb: np.ndarray) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid, tier="episodic", literal_surface=surface, aaak_index="",
        embedding=emb.tolist(), community_id=None, centrality=0.0, detail_level=2,
        pinned=False, stability=0.3, difficulty=0.0, last_reviewed=None,
        never_decay=False, never_merge=False, provenance=[{"session_id": "sess-x"}],
        created_at=now, updated_at=now, tags=["capture"], language="en",
    )


class _DispatchStubEmbedder:
    def __init__(self, vec: list[float]) -> None:
        self._vec = vec

    def embed(self, _text: str) -> list[float]:
        return list(self._vec)


def _build_dispatch_store(driver: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[MemoryStore, list[UUID], list[float]]:
    """A corpus sized past K_CANDIDATES, plus hebbian edges to a same-target
    neighbour beyond the ANN window and a two-hop chain -- so dispatch()'s
    own query_similar + hop1 + hop2 fetches all reach decode, not just the
    ANN seed.
    """
    _select_driver(driver, monkeypatch)
    store = MemoryStore(path=tmp_path / f"dispatch-{driver}")

    cue_vec = _dispatch_seeded_vec(1)
    target_id = uuid4()
    store.insert(_make_dispatch_rec(target_id, "the target record", cue_vec))
    ids = [target_id]
    for i in range(_DISPATCH_N):
        rid = uuid4()
        store.insert(_make_dispatch_rec(rid, f"filler {i}", _dispatch_seeded_vec(2000 + i)))
        ids.append(rid)
    flush_record_buffer(store)

    for other in ids[210:220]:
        store.boost_edges([(target_id, other)], edge_type="hebbian", delta=1.0)
    for a, b in zip(ids[220:225], ids[225:230]):
        store.boost_edges([(a, b)], edge_type="hebbian", delta=1.0)

    store._build_exact_index_sync()
    stub_embedder_for_store(monkeypatch, _DispatchStubEmbedder(cue_vec.tolist()))
    return store, ids, cue_vec.tolist()


def _dispatch_recall(store: MemoryStore, cue_vec: list[float]) -> dict:
    import iai_mcp.pipeline as _pm

    _pm._last_recall_latency_ms = 0.0
    return core_mod.dispatch(store, "memory_recall", {
        "cue": "dispatch decode-tier parity probe",
        "session_id": "dispatch-parity-test",
        "budget_tokens": 3000,
        "cue_embedding": cue_vec,
    })


# The fields the ephemeral graph payload builder in dispatch() actually
# reads off each candidate record (core/__init__.py set_node_payload call) --
# the same set test_dispatch_candidate_payload_keys_unchanged pins statically.
_DISPATCH_PAYLOAD_FIELDS = (
    "embedding", "literal_surface", "aaak_index", "created_at", "stability",
    "tier", "tags", "language",
)


def _dispatch_fetch_field_snapshot(
    store: MemoryStore, cue_vec: list[float], monkeypatch: pytest.MonkeyPatch,
) -> dict:
    """Snapshots _DISPATCH_PAYLOAD_FIELDS for every candidate returned by
    query_similar/get_batch during one dispatch() call -- the exact fields
    the ephemeral graph payload builder consumes, independent of whatever
    centrality/seed-pick/spread scoring later does with them. Comparing
    this snapshot across decode tiers isolates the rank-view decoder's
    correctness from the graph-ranking pipeline's own pre-existing,
    decode-tier-independent non-reproducibility (age-penalty wall clock,
    centrality on an ephemeral graph -- see module docstring).
    """
    snap: dict = {}

    def _record(rid, rec) -> None:
        snap[rid] = tuple(
            tuple(v) if isinstance(v, list) else str(v) if k == "created_at" else v
            for k, v in ((f, getattr(rec, f, None)) for f in _DISPATCH_PAYLOAD_FIELDS)
        )

    _orig_qs = _MS.query_similar
    _orig_gb = _MS.get_batch

    def _qs_wrap(self, *a, **kw):
        out = _orig_qs(self, *a, **kw)
        for _rec, _score in out:
            _record(_rec.id, _rec)
        return out

    def _gb_wrap(self, ids, *a, **kw):
        out = _orig_gb(self, ids, *a, **kw)
        for _rid, _rec in out.items():
            _record(_rid, _rec)
        return out

    monkeypatch.setattr(_MS, "query_similar", _qs_wrap)
    monkeypatch.setattr(_MS, "get_batch", _gb_wrap)
    _dispatch_recall(store, cue_vec)
    monkeypatch.setattr(_MS, "query_similar", _orig_qs)
    monkeypatch.setattr(_MS, "get_batch", _orig_gb)
    return snap


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_dispatch_candidate_fetch_fields_match_eager_lazy(
    driver, tmp_path, monkeypatch, _from_row_call_counter,
):
    """A dominant real-store decode-cost site: dispatch()'s own
    query_similar(k=K_CANDIDATES) plus hop1/hop2/rich-club get_batch.
    Field-level parity (not full-response equality -- see
    _dispatch_fetch_field_snapshot) between eager and lazy for every
    candidate either fetch site returns, with a full-_from_row call-count
    drop.
    """
    store, _ids, cue_vec = _build_dispatch_store(driver, tmp_path, monkeypatch)

    # _collect_graph_pool recomputes fresh every call (no cross-call
    # memoization) and its store-fallback resolution jitters by a small id
    # count call-to-call, independent of the decode-tier kill-switch --
    # confirmed via same-arm repeated calls during this test's development.
    # Retry bounded 5x, accepting the first attempt where both arms fetch
    # the identical candidate id set; a genuine decode-tier divergence
    # reproduces on every attempt, this background noise does not.
    eager_snap: dict = {}
    lazy_snap: dict = {}
    eager_from_row_calls = 0
    lazy_from_row_calls = 0
    for _attempt in range(1, 6):
        monkeypatch.setenv(LAZY_DECODE_KILL_SWITCH_ENV, "1")
        _from_row_call_counter.start_recall_phase()
        eager_snap = _dispatch_fetch_field_snapshot(store, cue_vec, monkeypatch)
        eager_from_row_calls = _from_row_call_counter.recall_total

        monkeypatch.delenv(LAZY_DECODE_KILL_SWITCH_ENV, raising=False)
        _from_row_call_counter.start_recall_phase()
        lazy_snap = _dispatch_fetch_field_snapshot(store, cue_vec, monkeypatch)
        lazy_from_row_calls = _from_row_call_counter.recall_total

        assert eager_snap, "harness self-check: eager arm fetched zero candidates"
        if set(lazy_snap.keys()) == set(eager_snap.keys()):
            break

    assert set(lazy_snap.keys()) == set(eager_snap.keys()), (
        "lazy and eager decode fetched a different candidate id set on "
        "every retry attempt -- a real decode-tier divergence, not the "
        "known pool-resolution jitter"
    )
    for rid in eager_snap:
        assert lazy_snap[rid] == eager_snap[rid], (
            f"candidate {rid}: lazy fields diverge from eager -- "
            f"lazy={lazy_snap[rid]!r} eager={eager_snap[rid]!r}"
        )

    assert lazy_from_row_calls < eager_from_row_calls, (
        f"lazy decode made {lazy_from_row_calls} full _from_row calls on "
        f"dispatch()'s own fetches, not fewer than eager's "
        f"{eager_from_row_calls}"
    )


def test_dispatch_candidate_payload_keys_unchanged():
    """Static shape guard: dispatch()'s ephemeral graph payload builder
    must keep reading exactly today's field set from each candidate. A new
    key here (e.g. salience_level, community_id, structure_hv) would start
    silently propagating a field the rank-view decoder defers to finalist
    hydration, on a path this plan's dynamic tests do not cover -- this
    guard catches that shape change directly.
    """
    src = inspect.getsource(core_mod)
    m = re.search(r"graph\.set_node_payload\(_rec\.id, \{(.*?)\}\)", src, re.DOTALL)
    assert m, "set_node_payload({...}) call not found in dispatch() -- guard scan went stale"
    keys = set(re.findall(r'"(\w+)":', m.group(1)))
    assert keys == {
        "embedding", "surface", "centrality", "tier", "tags", "language",
        "aaak_index", "created_at", "stability", "valence",
    }, f"dispatch() candidate payload key set changed: {keys}"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_dispatch_finalist_hydration_still_backfills_session_id(
    driver, tmp_path, monkeypatch, _frozen_age_penalty,
):
    """recall_for_response's existing _enrich_ids/get_batch finalist
    hydration must still fire and backfill session_id even though most
    candidates now arrive as rank-views with session_id=None -- it was
    written assuming only a graph-native minority needed it.
    """
    store, _ids, cue_vec = _build_dispatch_store(driver, tmp_path, monkeypatch)
    monkeypatch.delenv(LAZY_DECODE_KILL_SWITCH_ENV, raising=False)

    resp = _dispatch_recall(store, cue_vec)

    assert resp["hits"], "dispatch() returned no hits for the parity fixture"
    assert all(h.get("session_id") == "sess-x" for h in resp["hits"]), (
        "finalist hydration did not backfill session_id for every hit -- "
        f"got {[h.get('session_id') for h in resp['hits']]}"
    )
