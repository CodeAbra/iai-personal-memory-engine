"""End-to-end coverage for noise exclusion at the memory_recall serving path.

A record whose literal_surface carries a machine-traffic template-header
signature must never leak into the served hits or the rendered per-turn
recall block, even when ordinary ranking places it top-k -- unless the
record is locked (never_merge/never_decay), in which case it is still
served. The filter must never mutate stored text and must never raise.
"""
from __future__ import annotations

import hashlib
import math
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from iai_mcp.core import dispatch as core_dispatch
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord

_HOOKS_DIR = Path(__file__).resolve().parents[1] / "src/iai_mcp/_deploy/hooks"
if str(_HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_HOOKS_DIR))

from _recall_render import render_recall_block  # noqa: E402

# Matches the query-generation eval harness's own prompt template header
# already in the reused noise predicate.
NOISE_PREFIX = "Below is one record from a developer's personal memory system."


def _stable_seed(seed: str) -> int:
    # builtin hash() is randomized per process (no PYTHONHASHSEED pinned);
    # derive a fixed integer so vectors are reproducible across runs/seeds.
    return int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")


def _unit_vector(seed: str, dim: int = EMBED_DIM) -> list[float]:
    rng = random.Random(_stable_seed(seed))
    raw = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    mag = math.sqrt(sum(x * x for x in raw))
    return [x / mag for x in raw] if mag > 0 else raw


def _make_record(
    *,
    literal_surface: str,
    embedding: list[float],
    never_merge: bool = False,
    never_decay: bool = False,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=literal_surface,
        aaak_index="",
        embedding=embedding,
        community_id=None,
        centrality=0.5,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=never_decay,
        never_merge=never_merge,
        provenance=[],
        created_at=now,
        updated_at=now,
        language="en",
    )


def _fresh_store(tmp_path) -> MemoryStore:
    return MemoryStore(tmp_path, user_id="alice")


def test_top_ranked_noise_record_absent_from_hits_and_rendered_block(tmp_path):
    store = _fresh_store(tmp_path)
    noise_vec = _unit_vector("noise-record")
    noise_rec = _make_record(
        literal_surface=f"{NOISE_PREFIX} alice's session about dashboards.",
        embedding=noise_vec,
    )
    store.insert(noise_rec)
    flush_record_buffer(store)

    result = core_dispatch(store, "memory_recall", {
        "cue": "dashboards and sessions",
        "session_id": "noise-filter-probe",
        "cue_embedding": list(noise_vec),
    })

    hit_ids = [h["record_id"] for h in result["hits"]]
    assert str(noise_rec.id) not in hit_ids

    block = render_recall_block(result)
    assert "alice's session about dashboards" not in block


def test_exclude_serving_noise_skips_batch_lookup_when_nothing_flagged():
    from iai_mcp.core._serving_filter import exclude_serving_noise
    from iai_mcp.types import MemoryHit

    class _RaisingOnGetBatch:
        def get_batch(self, *args, **kwargs):
            raise AssertionError("get_batch must not run when no hit is flagged")

    hits = [
        MemoryHit(
            record_id=uuid.uuid4(),
            score=0.9,
            reason="test",
            literal_surface="alice prefers dark mode for the editor",
            adjacent_suggestions=[],
        ),
    ]
    assert exclude_serving_noise(_RaisingOnGetBatch(), hits) == hits


def test_legitimate_hit_order_unchanged_when_noise_record_removed(tmp_path):
    store = _fresh_store(tmp_path)
    cue_vec = _unit_vector("shared-cue-vector")

    noise_rec = _make_record(
        literal_surface=f"{NOISE_PREFIX} an eval harness artifact.",
        embedding=cue_vec,
    )
    close_rec = _make_record(
        literal_surface="alice's closest matching note about the topic",
        embedding=cue_vec,
    )
    far_rec = _make_record(
        literal_surface="alice's unrelated note about groceries",
        embedding=_unit_vector("unrelated-groceries"),
    )
    store.insert(noise_rec)
    store.insert(close_rec)
    store.insert(far_rec)
    flush_record_buffer(store)

    result = core_dispatch(store, "memory_recall", {
        "cue": "topic notes",
        "session_id": "noise-filter-probe",
        "cue_embedding": list(cue_vec),
    })

    hit_ids = [h["record_id"] for h in result["hits"]]
    assert str(noise_rec.id) not in hit_ids
    assert str(close_rec.id) in hit_ids
    close_pos = hit_ids.index(str(close_rec.id))
    if str(far_rec.id) in hit_ids:
        far_pos = hit_ids.index(str(far_rec.id))
        assert close_pos < far_pos


def test_never_merge_noise_record_is_still_served(tmp_path):
    store = _fresh_store(tmp_path)
    vec = _unit_vector("pinned-never-merge-noise")
    rec = _make_record(
        literal_surface=f"{NOISE_PREFIX} a pinned record via never_merge.",
        embedding=vec,
        never_merge=True,
    )
    store.insert(rec)
    flush_record_buffer(store)

    result = core_dispatch(store, "memory_recall", {
        "cue": "pinned record probe",
        "session_id": "noise-filter-probe",
        "cue_embedding": list(vec),
    })

    hit_ids = [h["record_id"] for h in result["hits"]]
    assert str(rec.id) in hit_ids


def test_never_decay_noise_record_is_still_served(tmp_path):
    store = _fresh_store(tmp_path)
    vec = _unit_vector("pinned-never-decay-noise")
    rec = _make_record(
        literal_surface=f"{NOISE_PREFIX} a pinned record via never_decay.",
        embedding=vec,
        never_decay=True,
    )
    store.insert(rec)
    flush_record_buffer(store)

    result = core_dispatch(store, "memory_recall", {
        "cue": "pinned record probe",
        "session_id": "noise-filter-probe",
        "cue_embedding": list(vec),
    })

    hit_ids = [h["record_id"] for h in result["hits"]]
    assert str(rec.id) in hit_ids


def _orthogonal_vector(seed: str, reference: list[float], dim: int = EMBED_DIM) -> list[float]:
    """A unit vector with (near) zero cosine to `reference`, so it neither
    ranks into the primary hit window by similarity nor gets confused with
    it -- the only path back into the response is the contradicts edge.

    Built via Gram-Schmidt (subtract the projection onto `reference`,
    renormalize) so the result is analytic, not sampled -- if the seeded
    draw were ever (near-)parallel to `reference`, fall back to a fixed
    basis axis so the result can never be degenerate."""
    raw = _unit_vector(seed, dim)
    dot = sum(a * b for a, b in zip(raw, reference))
    ortho = [a - dot * b for a, b in zip(raw, reference)]
    mag = math.sqrt(sum(x * x for x in ortho))
    if mag < 1e-9:
        axis_index = 0 if abs(reference[0]) < 0.9 else 1
        basis = [1.0 if i == axis_index else 0.0 for i in range(dim)]
        dot = sum(a * b for a, b in zip(basis, reference))
        ortho = [a - dot * b for a, b in zip(basis, reference)]
        mag = math.sqrt(sum(x * x for x in ortho))
    return [x / mag for x in ortho]


def _seed_anti_hit_pair(store, *, contradicting_literal_surface: str):
    """Seed a primary hit plus a contradicts-edge neighbour that is kept OUT
    of the primary hit window (orthogonal embedding + enough unrelated
    filler records + a tight token budget), so the neighbour can only reach
    the response through the anti_hits lane -- never by ranking as a hit
    itself. Without this separation an anti_hits assertion is vacuously
    true (the neighbour never surfaces anywhere, filtered or not)."""
    primary_vec = _unit_vector("anti-hit-primary-source")
    primary = _make_record(
        literal_surface="alice's release ships Tuesday",
        embedding=primary_vec,
    )
    contradicting = _make_record(
        literal_surface=contradicting_literal_surface,
        embedding=_orthogonal_vector("anti-hit-target", primary_vec),
    )
    store.insert(primary)
    store.insert(contradicting)
    for i in range(60):
        store.insert(_make_record(
            literal_surface=f"alice's unrelated filler note number {i}",
            embedding=_unit_vector(f"anti-hit-filler-{i}"),
        ))
    flush_record_buffer(store)

    store.boost_edges([(primary.id, contradicting.id)], edge_type="contradicts", delta=[1.0])

    import iai_mcp.retrieve as _retrieve
    import iai_mcp.runtime_graph_cache as _rgc
    _graph, _assignment, _rich_club = _retrieve.build_runtime_graph(store)
    _rgc.save(store, _assignment, _rich_club)

    return primary, contradicting


def _dispatch_anti_hit_probe(store, primary_vec: list[float]) -> dict:
    return core_dispatch(store, "memory_recall", {
        "cue": "alice's release status",
        "session_id": "noise-filter-probe",
        "cue_embedding": list(primary_vec),
        "budget_tokens": 40,
    })


def test_anti_hit_mechanism_surfaces_a_plain_contradicting_record(tmp_path):
    """Sanity companion: proves the seeding recipe actually exercises the
    anti_hits lane for a NON-noise record, so the noise-exclusion assertion
    below cannot pass vacuously."""
    store = _fresh_store(tmp_path)
    primary, contradicting = _seed_anti_hit_pair(
        store, contradicting_literal_surface="alice's release actually slipped",
    )

    result = _dispatch_anti_hit_probe(store, primary.embedding)

    hit_ids = [h["record_id"] for h in result["hits"]]
    anti_hit_ids = [h["record_id"] for h in result["anti_hits"]]
    assert str(primary.id) in hit_ids
    assert str(contradicting.id) not in hit_ids
    assert str(contradicting.id) in anti_hit_ids


def test_anti_hits_lane_filtered_independently_of_rendered_block(tmp_path):
    store = _fresh_store(tmp_path)
    primary, contradicting = _seed_anti_hit_pair(
        store,
        contradicting_literal_surface=f"{NOISE_PREFIX} alice's release actually slipped.",
    )

    result = _dispatch_anti_hit_probe(store, primary.embedding)

    anti_hit_ids = [h["record_id"] for h in result["anti_hits"]]
    assert str(contradicting.id) not in anti_hit_ids


def test_get_batch_returning_empty_keeps_the_noise_hit():
    from iai_mcp.core._serving_filter import exclude_serving_noise
    from iai_mcp.types import MemoryHit

    class _EmptyBatchStore:
        def get_batch(self, *args, **kwargs):
            return {}

    hit = MemoryHit(
        record_id=uuid.uuid4(), score=0.9, reason="test",
        literal_surface=f"{NOISE_PREFIX} an unresolved record.",
        adjacent_suggestions=[],
    )
    assert exclude_serving_noise(_EmptyBatchStore(), [hit]) == [hit]


def test_get_batch_raising_keeps_the_noise_hit_and_does_not_propagate():
    from iai_mcp.core._serving_filter import exclude_serving_noise
    from iai_mcp.types import MemoryHit

    class _RaisingBatchStore:
        def get_batch(self, *args, **kwargs):
            raise RuntimeError("synthetic lookup failure")

    hit = MemoryHit(
        record_id=uuid.uuid4(), score=0.9, reason="test",
        literal_surface=f"{NOISE_PREFIX} a record whose lookup blows up.",
        adjacent_suggestions=[],
    )
    assert exclude_serving_noise(_RaisingBatchStore(), [hit]) == [hit]


def test_surviving_hit_and_stored_literal_surface_are_byte_identical(tmp_path):
    """Byte-identity through the mutating-risk path: the record matches the
    noise signature (so it flows through the string prefilter AND the
    get_batch/is_excludable_noise resolve), is served only because it is
    carve-out protected, and its literal_surface must round-trip untouched."""
    store = _fresh_store(tmp_path)
    vec = _unit_vector("write-once-legit-record")
    text = f"{NOISE_PREFIX} a pinned record that must stay byte-identical after serving."
    rec = _make_record(literal_surface=text, embedding=vec, never_merge=True)
    store.insert(rec)
    flush_record_buffer(store)

    result = core_dispatch(store, "memory_recall", {
        "cue": "byte identical probe",
        "session_id": "noise-filter-probe",
        "cue_embedding": list(vec),
    })

    hit = next(h for h in result["hits"] if h["record_id"] == str(rec.id))
    assert hit["literal_surface"] == text
    stored = store.get(rec.id)
    assert stored.literal_surface == text


def test_mid_string_noise_phrase_is_served_not_substring_matched(tmp_path):
    store = _fresh_store(tmp_path)
    vec = _unit_vector("mid-string-noise-phrase")
    rec = _make_record(
        literal_surface=f"alice quoted a phrase mid-note: {NOISE_PREFIX} later on.",
        embedding=vec,
    )
    store.insert(rec)
    flush_record_buffer(store)

    result = core_dispatch(store, "memory_recall", {
        "cue": "mid string probe",
        "session_id": "noise-filter-probe",
        "cue_embedding": list(vec),
    })

    hit_ids = [h["record_id"] for h in result["hits"]]
    assert str(rec.id) in hit_ids


def test_crisis_mode_excludes_noise_but_serves_pinned_noise_record(tmp_path, monkeypatch):
    import iai_mcp.core as _core

    class _FixedEmbedder:
        def __init__(self, vec: list[float]) -> None:
            self._vec = vec

        def embed(self, _text: str) -> list[float]:
            return list(self._vec)

    _core._CRISIS_STATE_CACHE.clear()
    try:
        store = _fresh_store(tmp_path)
        target_vec = _unit_vector("crisis-path-target")

        noise_rec = _make_record(
            literal_surface=f"{NOISE_PREFIX} a crisis-path noise record.",
            embedding=target_vec,
        )
        pinned_noise_rec = _make_record(
            literal_surface=f"{NOISE_PREFIX} a crisis-path pinned noise record.",
            embedding=target_vec,
            never_merge=True,
        )
        store.insert(noise_rec)
        store.insert(pinned_noise_rec)
        flush_record_buffer(store)

        monkeypatch.setattr(
            "iai_mcp.lifecycle_state.load_state",
            lambda *_a, **_kw: {"crisis_mode": True, "current_state": "SLEEP"},
        )
        monkeypatch.setattr(
            "iai_mcp.embed.embedder_for_store",
            lambda _store: _FixedEmbedder(target_vec),
        )

        result = core_dispatch(store, "memory_recall", {
            "cue": "crisis mode probe",
            "session_id": "noise-filter-probe",
        })

        assert result["_degraded"] is True
        hit_ids = [h["record_id"] for h in result["hits"]]
        assert str(noise_rec.id) not in hit_ids
        assert str(pinned_noise_rec.id) in hit_ids
    finally:
        _core._CRISIS_STATE_CACHE.clear()


def test_crisis_mode_filter_failure_serves_candidate_not_empty(tmp_path, monkeypatch):
    """Fail-toward-leak on the crisis-degrade path: a broken noise predicate
    must serve the candidate, matching the normal path's guard, and must
    never propagate to the tier's last-resort empty response."""
    import iai_mcp.core as _core

    class _FixedEmbedder:
        def __init__(self, vec: list[float]) -> None:
            self._vec = vec

        def embed(self, _text: str) -> list[float]:
            return list(self._vec)

    _core._CRISIS_STATE_CACHE.clear()
    try:
        store = _fresh_store(tmp_path)
        target_vec = _unit_vector("crisis-path-filter-failure")

        rec = _make_record(
            literal_surface="alice's crisis-path record that must survive a filter failure",
            embedding=target_vec,
        )
        store.insert(rec)
        flush_record_buffer(store)

        monkeypatch.setattr(
            "iai_mcp.lifecycle_state.load_state",
            lambda *_a, **_kw: {"crisis_mode": True, "current_state": "SLEEP"},
        )
        monkeypatch.setattr(
            "iai_mcp.embed.embedder_for_store",
            lambda _store: _FixedEmbedder(target_vec),
        )
        monkeypatch.setattr(
            "iai_mcp.core._serving_filter.is_excludable_noise",
            lambda _rec: (_ for _ in ()).throw(RuntimeError("synthetic filter failure")),
        )

        result = core_dispatch(store, "memory_recall", {
            "cue": "crisis mode filter failure probe",
            "session_id": "noise-filter-probe",
        })

        assert result["_degraded"] is True
        hit_ids = [h["record_id"] for h in result["hits"]]
        assert str(rec.id) in hit_ids
    finally:
        _core._CRISIS_STATE_CACHE.clear()


def test_all_noise_hits_report_arousal_failure_not_success(tmp_path):
    """The arousal signal must reflect POST-filter hits: a cue whose only
    matches are noise records reports a miss, not a false success from
    pre-filter hit presence."""
    import iai_mcp.core as _core

    store = _fresh_store(tmp_path)
    vec = _unit_vector("arousal-all-noise-cue")
    noise_rec = _make_record(
        literal_surface=f"{NOISE_PREFIX} the only match for this cue.",
        embedding=vec,
    )
    store.insert(noise_rec)
    flush_record_buffer(store)

    def _snapshot():
        state = _core._arousal_state
        return (0, 0) if state is None else (state.success_count, state.error_count)

    before = _snapshot()

    result = core_dispatch(store, "memory_recall", {
        "cue": "all noise probe",
        "session_id": "noise-filter-probe",
        "cue_embedding": list(vec),
    })

    after = _snapshot()

    assert result["hits"] == []
    assert after[0] == before[0]
    # >= rather than exact +1: a fallback path can also fire the unrelated
    # "error" event, which shares the same counter -- the load-bearing
    # assertion is that a miss registers at all, not the exact delta.
    assert after[1] >= before[1] + 1
